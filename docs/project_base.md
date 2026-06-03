# Codebase Intelligence Assistant - Project Base

Date: 2026-06-03

This document is the working base for the project report, slides, and demo. It explains the goal, the complete pipeline, the methods and technologies used, the experiments we ran, the evaluation metrics, current results, limitations, and possible improvements.

## 1. Executive Summary

The Codebase Intelligence Assistant is a local-first retrieval-augmented generation system for answering questions about a software repository. The system ingests a repository, extracts semantic code chunks, builds dense and sparse retrieval indexes, retrieves relevant snippets for a user question, generates a grounded answer with a local code model, and shows the answer plus retrieved citations in a Streamlit chat UI.

The current demo corpus is FastAPI. The active index was built from:

```text
repo_path: data/raw/fastapi
indexed paths: fastapi, README.md
chunk_count: 526
dense model: microsoft/graphcodebert-base
dense store: index/chroma_db
sparse store: index/bm25/bm25.pkl
```

The best default UI configuration based on the latest ablation plus DeepEval LLM-judge evaluation is:

```python
iterative_rag(prompt, mode="single", confidence=False)
```

This corresponds to condition B: hybrid retrieval with one generation pass. It has the best final answer quality in the DeepEval results, with answer correctness `0.700`, while condition D remains useful as an optional "Deep search" mode because it has the best retrieval coverage and slightly higher faithfulness.

## 2. Problem and Motivation

Answering questions about a codebase is hard because source code is large, split across many files, and often uses internal names that users do not know. A useful codebase assistant must:

- find relevant code even when the question uses natural language rather than exact identifiers;
- handle exact function, class, and file names when they are present;
- cite retrieved source snippets so answers can be checked;
- refuse when the answer is not present in the retrieved code;
- expose enough context for debugging and evaluation;
- run locally for privacy and reproducibility.

The project therefore combines dense semantic retrieval, sparse lexical retrieval, rank fusion, local LLM generation, grounding checks, confidence experiments, and an evaluation harness.

## 3. Repository Layout

```text
codebase-intelligence-assistant/
  app/
    streamlit_app.py          Streamlit chat UI.
  data/
    raw/                      Cloned or local repositories, gitignored.
    chunks/                   Embedding cache, gitignored.
  docs/
    project_base.md           This document.
  eval/
    run_eval.py               Manual evaluation harness.
    ablation.py               T12 ablation runner.
    deepeval_judge.py         DeepEval GEval LLM-as-judge evaluator.
    test_set.json             Original local-project eval set.
    test_set_fastapi.json     FastAPI eval set used for current results.
  index/
    chroma_db/                Chroma dense vector store, gitignored.
    bm25/bm25.pkl             BM25 sparse index, gitignored.
    manifest.json             Index manifest, gitignored.
  results/
    ablation_fastapi.csv      Latest FastAPI no-confidence ablation table.
    ablation_fastapi.json     Full latest FastAPI no-confidence ablation.
    deepeval_fastapi.csv      Latest DeepEval judge summary.
    deepeval_fastapi.json     Full latest DeepEval judge records.
    *_runs.jsonl              Per-query retrieval/generation traces.
    *_confidence_log.jsonl    Confidence logs from confidence-enabled runs.
  src/
    schema.py                 Shared Chunk dataclass.
    ingest.py                 Repository ingestion.
    chunker.py                Semantic chunk extraction.
    build_index.py            Combined dense + BM25 index builder.
    index_dense.py            GraphCodeBERT + Chroma dense retrieval.
    bm25_index.py             BM25 sparse retrieval.
    retriever.py              Hybrid retrieval with RRF.
    generator.py              Ollama/Qwen grounded generation.
    pipeline.py               Single/iterative RAG orchestration.
    confidence.py             HonestCoder-style confidence estimation.
    grounding.py              Identifier grounding check.
  tests/
    test_*.py                 Unit tests for the full pipeline.
```

## 4. High-Level Pipeline

The end-to-end flow is:

```text
Repository
  -> Ingestion
  -> Semantic chunking
  -> Dense index: GraphCodeBERT embeddings in Chroma
  -> Sparse index: BM25 over identifier-aware tokens
  -> Hybrid retrieval: Reciprocal Rank Fusion
  -> RAG pipeline: single-pass or iterative two-pass retrieval
  -> Local generation: Ollama OpenAI-compatible API with Qwen2.5-Coder
  -> Grounding/confidence layers
  -> Streamlit UI with answer and retrieved snippets
  -> Evaluation, ablation, and DeepEval judge results
```

The current UI uses the default pipeline retriever, which is hybrid search. The default mode is "Fast answer", which maps to `mode="single"`. The optional "Deep search" mode maps to `mode="iterative"`. With `confidence=False`, the answer is not overwritten by the experimental confidence refusal layer.

## 5. Shared Data Model

The central unit is the `Chunk` dataclass in `src/schema.py`.

```python
@dataclass(frozen=True)
class Chunk:
    id: str
    filepath: str
    function_name: str
    start_line: int
    end_line: int
    docstring: str
    source: str
    language: str
```

Each chunk represents a semantic unit of code or documentation. The key fields are:

| Field | Meaning |
|---|---|
| `id` | Stable hash derived from file path, symbol name, and line range. |
| `filepath` | Repository-relative path using POSIX separators. |
| `function_name` | Function, class, method, module, or README section name. |
| `start_line` / `end_line` | Source line interval used for citations. |
| `docstring` | Extracted docstring or README heading text. |
| `source` | Raw source text for retrieval and generation. |
| `language` | Language label, currently mostly `python`, `markdown`, or text. |

`Chunk.to_dict()` returns a JSON-serializable dictionary used by retrieval, generation, UI display, and evaluation.

## 6. Ingestion

File: `src/ingest.py`

The ingestion stage collects raw repository files before chunking. It supports:

- local repository paths;
- Git clone URLs through GitPython;
- Python, JavaScript, TypeScript, Java, and README-like files;
- JSON persistence of the flat raw file list.

Important behavior:

- Supported source extensions: `.py`, `.js`, `.ts`, `.java`.
- README suffixes: no suffix, `.md`, `.rst`, `.txt`.
- Skipped directories include `.git`, `.venv`, `node_modules`, `dist`, `build`, `coverage`, `migrations`, `__pycache__`, and other cache folders.
- License and notice files are skipped.
- Binary or non-UTF-8 files are skipped.
- Generated files are skipped based on filename suffixes and header markers such as `@generated`, `auto-generated`, `do not edit`, and `generated by`.
- Files over the line limit are skipped. The default is 500 lines.

Useful commands:

```bash
.venv/bin/python -m src.ingest --repo tests/fixtures/sample_python_project
.venv/bin/python -m src.ingest --url https://github.com/example/project.git
.venv/bin/python -m src.ingest --repo tests/fixtures/sample_python_project --json
.venv/bin/python -m src.ingest --repo tests/fixtures/sample_python_project --output data/raw/files.json
```

In the current FastAPI demo, the repository is already present under `data/raw/fastapi`.

## 7. Semantic Chunking

File: `src/chunker.py`

The chunker converts repository files into semantic `Chunk` records. It is Python-first and README-aware.

### 7.1 Python Symbol Extraction

The chunker extracts:

- classes;
- methods;
- functions;
- async functions;
- module-level imports/constants as `__module__` chunks.

It uses tree-sitter for symbol boundaries when available and Python `ast` for docstring extraction. There is also an AST-based fallback for local tests and environments where tree-sitter dependencies are not installed.

### 7.2 README Chunking

README files are split into section-like chunks. README chunks use function names like:

```text
README:standard-dependencies
README:about-fastapi-cloud
README:interactive-api-docs-upgrade
```

This lets documentation participate in the same retrieval/evaluation path as code.

### 7.3 Stable IDs

Chunk IDs are deterministic:

```text
sha1(filepath:function_name:start_line:end_line)[:16]
```

Stable IDs matter for caching, deduplication, and reproducible retrieval.

### 7.4 Oversized Chunk Splitting

Large symbols can be split into overlapping child chunks using:

```text
max_chunk_lines = 220
chunk_overlap_lines = 20
```

The split parts keep the original `filepath` and `function_name`, which lets dense retrieval attach neighboring chunks from the same symbol later. This matters for large classes like `FastAPI` and large route handling functions.

Useful command:

```bash
.venv/bin/python -m src.chunker --repo data/raw/fastapi --json
```

## 8. Index Building

File: `src/build_index.py`

The combined builder is the recommended indexing path because it builds dense and sparse indexes from the same filtered chunk corpus. That prevents dense and BM25 from silently searching different code.

Current FastAPI index command:

```bash
.venv/bin/python -m src.build_index \
  --repo data/raw/fastapi \
  --include-path-prefix fastapi \
  --include-path-prefix README.md \
  --exclude-tests
```

Current manifest summary:

```text
repo_path: data/raw/fastapi
chunk_count: 526
include_path_prefixes: fastapi, README.md
exclude_tests: true
dense model: microsoft/graphcodebert-base
embedding_cache_path: data/chunks/embeddings.pkl
chroma_path: index/chroma_db
bm25_path: index/bm25/bm25.pkl
```

The manifest is important because it records exactly which repository and path prefixes are indexed. This makes evaluation runs reproducible and prevents interpreting metrics without knowing the active corpus.

## 9. Dense Retrieval

File: `src/index_dense.py`

Dense retrieval uses GraphCodeBERT and ChromaDB:

- model: `microsoft/graphcodebert-base`;
- tokenizer/model loaded through HuggingFace Transformers;
- embeddings computed with mean pooling over token states;
- vectors normalized with PyTorch;
- vectors stored in ChromaDB under `index/chroma_db`;
- embedding cache stored in `data/chunks/embeddings.pkl`.

### 9.1 Embedding Text

Each chunk is embedded using a structured text representation:

```text
path: <filepath>
symbol: <function_name>
language: <language>
docstring: <docstring, if present>
source:
<raw source code>
```

This helps the embedding model use file path, symbol name, documentation, and source content.

### 9.2 Caching

Embeddings are cached by:

- model name;
- chunk IDs;
- hash of each chunk embedding text;
- cache version.

This prevents recomputing embeddings when the indexed corpus has not changed.

### 9.3 Chroma Metadata

Each vector stores metadata:

- `filepath`;
- `function_name`;
- `start_line`;
- `end_line`;
- `docstring`;
- `language`;
- `model_name`.

The generator and UI use this metadata for citations and snippet display.

### 9.4 Linked Neighbor Chunks

For large symbols split into parts, dense retrieval can attach neighboring chunks from the same symbol using `linked_window`. This helps avoid retrieving only the middle of a long function or class without adjacent context.

## 10. Sparse Retrieval with BM25

File: `src/bm25_index.py`

BM25 retrieval uses `rank_bm25`. Its purpose is to handle exact identifiers and lexical matches, which dense retrieval often misses.

### 10.1 Identifier-Aware Tokenization

The tokenizer:

- splits acronym boundaries;
- splits camelCase;
- splits snake_case and punctuation;
- lowercases tokens;
- removes natural-language stopwords from queries;
- expands some query words, for example `handled`, `handles`, and `handling` to `handle` and `handler`.

Examples:

```text
processPayment -> process, payment
get_user_by_id -> get, user, by, id
HTTPSConnection -> https, connection
```

### 10.2 BM25 Documents

BM25 scores a concatenation of:

- function name;
- docstring;
- source;
- file path.

The index is persisted with pickle to `index/bm25/bm25.pkl`.

## 11. Hybrid Retrieval with RRF

File: `src/retriever.py`

Hybrid retrieval calls both:

- dense retrieval from Chroma/GraphCodeBERT;
- sparse retrieval from BM25.

It then merges the ranked lists with Reciprocal Rank Fusion.

Formula:

```text
score(chunk) = sum(1 / (k + rank_i))
```

where `k=60` by default and `rank_i` is the 1-based position of the chunk in each retrieval list.

Why RRF helps:

- Dense retrieval finds semantically related code even when exact names differ.
- BM25 finds exact identifiers, file names, and method names.
- RRF gives a high score to chunks that appear in both lists, while still preserving useful results from only one retriever.

The hybrid search fetches `top_k * dense_k_multiplier` candidates from both retrievers, fuses them, and returns the final top `k`.

Current default for user-facing retrieval:

```text
top_k: 7
retriever: hybrid_search
linked_window: 1
```

## 12. Grounded Generation

File: `src/generator.py`

Generation uses Ollama through the OpenAI-compatible API:

```text
base URL: http://localhost:11434/v1
model: qwen2.5-coder:7b
api key default: ollama
```

The generator receives retrieved chunks and the user question. It formats each chunk as a numbered citation:

```text
[1] fastapi/openapi/docs.py::get_swagger_ui_html lines 40-194
Docstring: ...
Source:
...
```

The system prompt requires:

- answer strictly using provided source snippets;
- say `I cannot find this in the codebase.` when unsupported;
- do not invent names, libraries, or logic;
- cite every factual sentence with snippet labels like `[1]`;
- answer in 2-6 concise bullets;
- say what can and cannot be determined if context is partial.

The generator includes additional guardrails:

- no context means immediate refusal;
- empty model answer means refusal;
- missing citations means refusal;
- citations outside the context range mean refusal;
- if the query contains required code-like identifiers, the final answer must mention them.

This makes the system conservative, which is good for trust but can also reduce recall if the prompt or retrieved context is imperfect.

## 13. RAG Orchestration

File: `src/pipeline.py`

The main public function is:

```python
iterative_rag(query, mode="single", top_k=7, confidence=False)
```

It returns a dictionary with:

- `answer`;
- `retrieved_chunks`;
- `trace`;
- `mode`;
- optional `partial_answer`;
- optional confidence fields;
- grounding status and ungrounded claims.

### 13.1 Single-Pass Mode

Single mode:

```text
query -> hybrid_search(query) -> generate(context, query)
```

This is the current UI default because it gave the best DeepEval answer correctness in the FastAPI judge run.

### 13.2 Iterative Two-Pass Mode

Iterative mode:

```text
Pass 1:
query -> hybrid_search(query) -> partial answer

Pass 2:
query + partial answer -> hybrid_search(refined query) -> final answer
```

This is inspired by RepoCoder-style iterative retrieval. It can improve retrieval because the partial answer introduces useful terms that were absent from the original query.

In the latest FastAPI results, hybrid iterative had the best context hit rate and slightly higher faithfulness, but hybrid single-pass had the best DeepEval answer correctness. Therefore:

- use hybrid single-pass as the default UI mode;
- use hybrid iterative as optional "Deep search" when the goal is maximum retrieval coverage or deeper source inspection.

### 13.3 Logging

The pipeline logs retrieval and generation traces to JSONL files, including:

- query;
- retrieved chunks;
- pass 1 and pass 2 traces;
- generation latency;
- retrieval latency;
- final answer.

These logs are useful for debugging and for showing concrete before/after retrieval behavior in presentations.

## 14. Grounding Check Layer

File: `src/grounding.py`

After generation, the grounding check extracts possible code claims from the answer:

- backticked identifiers;
- function calls;
- snake_case;
- camelCase;
- PascalCase.

It then checks whether each extracted claim appears in the retrieved chunks' source, function names, or file paths. If not, it reports the claim as ungrounded.

The result shape is:

```json
{
  "grounded": true,
  "ungrounded_claims": []
}
```

This is a lightweight hallucination detector. It is intentionally simple and fast. It can produce false positives or miss natural-language hallucinations, but it is useful for catching invented function or class names.

## 15. HonestCoder Confidence Layer

File: `src/confidence.py`

The confidence layer is optional. It estimates answer reliability by sampling multiple answers and measuring agreement.

Default temperatures:

```text
0.0, 0.5, 0.9
```

Scores used:

- lexical similarity: token overlap between answers;
- semantic similarity: cosine similarity from `sentence-transformers/all-MiniLM-L6-v2`;
- combined score: `0.4 * lexical + 0.6 * semantic`.

Default threshold:

```text
0.72
```

If the deterministic answer is already a refusal, or if agreement is below threshold, the pipeline can replace the answer with:

```text
I cannot find this in the codebase.
```

### 15.1 What We Learned

The confidence layer is useful as an experiment and for logging uncertainty, but it was too conservative in the FastAPI UI setup. It caused many answerable questions to be refused. Based on ablation:

- keep confidence available for experiments;
- do not enable it by default in the UI;
- tune thresholds before using it as a production refusal layer.

## 16. Streamlit UI

File: `app/streamlit_app.py`

The UI is a Streamlit chat application. It:

- accepts a natural-language codebase question;
- calls the RAG pipeline;
- defaults to the DeepEval-backed "Fast answer" mode;
- exposes an optional "Deep search" mode for higher retrieval coverage;
- shows the selected mode's DeepEval summary when judge results are available;
- shows the answer;
- stores chat history in `st.session_state`;
- displays retrieved snippets in an expander;
- shows line numbers, file paths, and function/class names;
- shows grounding warnings when the grounding layer flags ungrounded claims;
- can show confidence details if the confidence layer is enabled.

Default UI call:

```python
result = iterative_rag(prompt, mode="single", confidence=False)
```

Optional deep-search call:

```python
result = iterative_rag(prompt, mode="iterative", confidence=False)
```

Reason for the default:

- this maps to hybrid single-pass retrieval;
- it had the best DeepEval answer correctness score in the latest judge run;
- it keeps the lower latency of one retrieval/generation pass;
- it avoids the over-refusal observed with confidence enabled.

Run the UI:

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

If `streamlit` is not installed in the environment, install it or run through the project requirements if included by the local environment.

## 17. Evaluation Harness

File: `eval/run_eval.py`

The evaluation harness accepts any pipeline function shaped like:

```python
pipeline_fn(question) -> answer_or_result
```

The pipeline may return either a string or a dictionary with:

- `answer`;
- `retrieved_chunks`, `retrieved_contexts`, `contexts`, or `citations`.

The harness normalizes the output and scores it against a manual JSON test set.

### 17.1 Test Set Format

Each item has:

```json
{
  "id": "fastapi-swagger-ui-parameters",
  "question": "How does FastAPI apply custom Swagger UI parameters when generating the docs HTML?",
  "ground_truth_answer": "...",
  "ground_truth_context": [
    {
      "filepath": "fastapi/openapi/docs.py",
      "function_name": "get_swagger_ui_html"
    }
  ],
  "expected_refusal": false
}
```

Current test sets:

| File | Items | Purpose |
|---|---:|---|
| `eval/test_set.json` | 19 | Original questions about this assistant repository. |
| `eval/test_set_fastapi.json` | 20 | FastAPI questions aligned with the current FastAPI index. |

### 17.2 Metrics

| Metric | Meaning |
|---|---|
| `total_questions` | Total number of questions in the test set. |
| `answerable_questions` | Questions expected to have source-supported answers. |
| `refusal_questions` | Questions expected to be out of scope. |
| `context_hit_rate` | For answerable questions, fraction where at least one expected context was retrieved. |
| `context_recall` | Fraction of expected contexts retrieved per answerable question, averaged. |
| `context_precision` | Fraction of retrieved contexts that match expected contexts, averaged. |
| `answer_relevance_proxy` | Token overlap between answer and ground-truth answer. This is a lightweight proxy, not semantic grading. |
| `refusal_accuracy` | For expected refusals, fraction of answers that contain refusal markers. |
| `adversarial_refusal_rate` | For extra adversarial queries, fraction correctly refused or flagged. |
| `avg_confidence` | Mean confidence score when confidence is enabled. Blank when disabled. |
| `low_confidence_rate` | Fraction low-confidence when confidence is enabled. Blank when disabled. |

### 17.3 DeepEval GEval LLM Judge

File: `eval/deepeval_judge.py`

The project now uses the professor-recommended DeepEval + GEval LLM-as-judge technique for final answer quality. The ablation metrics answer "Did retrieval find the right context?". DeepEval answers "Was the generated answer semantically correct and supported?".

The judge reads existing ablation outputs rather than rerunning the RAG pipeline:

```text
input test set: eval/test_set_fastapi.json
input ablation: results/ablation_fastapi.json
output JSON: results/deepeval_fastapi.json
output CSV: results/deepeval_fastapi.csv
judge model: qwen2.5-coder:7b through local Ollama
base URL: http://localhost:11434/v1/
```

Each record becomes a DeepEval test case:

```python
LLMTestCase(
    input=question,
    actual_output=answer,
    expected_output=ground_truth_answer,
    retrieval_context=retrieved_context_summaries,
)
```

Judge metrics:

| Metric | Meaning |
|---|---|
| `answer_correctness` | GEval score for whether the answer correctly answers the question compared with the ground truth, allowing paraphrases. |
| `faithfulness` | GEval score for whether the answer is supported by the retrieved context. |
| `refusal_quality` | GEval score for expected-refusal questions only. |
| `rouge1_f1` | Lexical ROUGE-1 F1 baseline for comparison only. |
| `rougeL_f1` | Lexical ROUGE-L F1 baseline for comparison only. |
| `bleu` | Lexical BLEU-style baseline for comparison only. |
| `error_count` | Number of judge calls that failed for a condition. |

The lexical scores are included only as baselines. The main report interpretation should use `answer_correctness`, `faithfulness`, and `refusal_quality`.

### 17.4 RAGAS

The harness has optional RAGAS integration for:

- faithfulness;
- context precision;
- context recall;
- answer relevancy.

RAGAS was considered, but DeepEval GEval was chosen for the current project because it matches the professor notebook and avoids the RAGAS/LangChain dependency issue seen in this environment. RAGAS remains optional research work, not the main evaluation method.

## 18. Ablation Experiments

File: `eval/ablation.py`

The T12 ablation compares four conditions:

| Condition | Retrieval | Retrieval passes |
|---|---|---|
| A | Dense only | Single-pass |
| B | Hybrid | Single-pass |
| C | Dense only | Iterative two-pass |
| D | Hybrid | Iterative two-pass |

The ablation also runs adversarial queries:

- the expected-refusal questions from the test set;
- `Which npm script deploys the React frontend to production?`;
- `Where is the Stripe webhook signature verified?`;
- `How does the Kubernetes operator rotate TLS certificates?`.

### 18.1 Commands

Validate the FastAPI test set:

```bash
.venv/bin/python -m eval.run_eval --test-set eval/test_set_fastapi.json
```

Run the current recommended ablation without confidence:

```bash
.venv/bin/python -m eval.ablation \
  --test-set eval/test_set_fastapi.json \
  --json-output results/ablation_fastapi.json \
  --csv-output results/ablation_fastapi.csv \
  --pipeline-log-path results/ablation_fastapi_runs.jsonl \
  --confidence-log-path results/ablation_fastapi_confidence_log.jsonl \
  --no-confidence
```

Run with confidence for research only:

```bash
.venv/bin/python -m eval.ablation \
  --test-set eval/test_set_fastapi.json \
  --json-output results/ablation_fastapi_confidence.json \
  --csv-output results/ablation_fastapi_confidence.csv \
  --pipeline-log-path results/ablation_fastapi_confidence_runs.jsonl \
  --confidence-log-path results/ablation_fastapi_confidence_log.jsonl
```

## 19. Current FastAPI Results

Latest retrieval result file:

```text
results/ablation_fastapi.csv
generated_at: 2026-06-03T15:20:17.943483+00:00
test_set: eval/test_set_fastapi.json
confidence_enabled: false
top_k: 7
```

### 19.1 No-Confidence FastAPI Ablation

| Condition | Retrieval | Passes | Context hit | Context recall | Context precision | Answer overlap | Refusal accuracy | Adversarial refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | Dense only | Single | 0.000 | 0.000 | 0.000 | 0.117 | 1.000 | 1.000 |
| B | Hybrid | Single | 0.737 | 0.737 | 0.120 | 0.488 | 1.000 | 1.000 |
| C | Dense only | Iterative | 0.053 | 0.053 | 0.008 | 0.105 | 1.000 | 1.000 |
| D | Hybrid | Iterative | 0.895 | 0.895 | 0.143 | 0.362 | 1.000 | 1.000 |

### 19.2 DeepEval Answer-Quality Results

Latest judge result files:

```text
results/deepeval_fastapi.csv
results/deepeval_fastapi.json
generated_at: 2026-06-03T16:48:12.334835+00:00
judge: DeepEval GEval
model: qwen2.5-coder:7b
base_url: http://localhost:11434/v1/
```

| Condition | Retrieval | Passes | Answer correctness | Faithfulness | Refusal quality | ROUGE-1 F1 | ROUGE-L F1 | BLEU | Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Dense only | Single | 0.079 | 0.025 | 0.800 | 0.101 | 0.075 | 0.055 | 0 |
| B | Hybrid | Single | 0.700 | 0.260 | 0.800 | 0.297 | 0.198 | 0.223 | 0 |
| C | Dense only | Iterative | 0.089 | 0.010 | 0.800 | 0.088 | 0.074 | 0.037 | 0 |
| D | Hybrid | Iterative | 0.526 | 0.280 | 0.800 | 0.218 | 0.157 | 0.150 | 0 |

### 19.3 Interpretation

The results support these conclusions:

- Hybrid retrieval is essential. Conditions B and D strongly outperform dense-only conditions A and C.
- Hybrid iterative retrieval is best for context coverage. Condition D found expected contexts for 17 of 19 answerable FastAPI questions.
- Hybrid single-pass is best for final answer quality. Condition B scored `0.700` DeepEval answer correctness, the best score by a clear margin.
- Hybrid iterative is slightly best for faithfulness. Condition D scored `0.280` faithfulness versus B at `0.260`, but its answer correctness was lower.
- The older token-overlap metric agrees directionally with DeepEval, but DeepEval is the better report metric because it allows paraphrases and judges semantic correctness.
- Dense-only retrieval is weak for this codebase/test set. It often retrieves semantically related but wrong README or framework-level snippets.
- Adversarial refusal is strong. All four conditions correctly refused or flagged all adversarial queries.
- DeepEval judge stability was good in this run. All four conditions had `error_count = 0`.
- Confidence should remain disabled in the UI for now. It over-refused answerable questions in previous experimentation.

### 19.4 UI Decision From Results

Use condition B as the default UI mode:

```python
iterative_rag(prompt, mode="single", confidence=False)
```

Why condition B:

- best DeepEval answer correctness;
- best lexical baselines among all conditions;
- still uses hybrid retrieval;
- perfect adversarial refusal in current evaluation;
- faster than iterative mode because it runs one retrieval/generation pass.

Expose condition D as an optional "Deep search" mode when:

- retrieval coverage matters more than answer conciseness;
- debugging or inspecting source;
- the user wants the mode with the best context hit rate and slightly higher faithfulness.

## 20. What We Tried and What Changed

### 20.1 DeepEval Instead of RAGAS as the Main Judge

We considered RAGAS because it is a common RAG evaluation library, and an optional RAGAS path exists in `eval/run_eval.py`. For the current report, we chose DeepEval GEval instead.

Why DeepEval was chosen:

- it matches the professor-recommended notebook;
- it directly grades generated answer quality;
- it works with a local Ollama judge model;
- it avoids relying on the unstable RAGAS/LangChain dependency stack for the main report;
- it complements the existing ablation metrics instead of replacing them.

### 20.2 Confidence-Enabled Runs

We tried the HonestCoder confidence layer as a refusal filter. It generated multiple answer variants and measured agreement. It successfully identified uncertainty, but in the UI setting it was too conservative and replaced too many potentially useful answers with refusals.

Decision:

- keep confidence as an optional research/evaluation feature;
- disable confidence in the UI;
- show confidence only if explicitly enabled.

### 20.3 RAGAS

RAGAS support was wired into `eval/run_eval.py`, but the installed LangChain/RAGAS stack reported a missing optional import involving `langchain_community.chat_models.vertexai`.

Decision:

- do not use RAGAS as the main report metric for this project;
- keep the optional path as research scaffolding;
- use DeepEval GEval for answer correctness, faithfulness, and refusal quality.

### 20.4 Dense vs Hybrid

Dense-only retrieval was tested because GraphCodeBERT should capture semantic similarity. In practice, for FastAPI questions with specific methods and classes, dense-only retrieval missed many exact targets.

Hybrid retrieval fixed this by adding BM25 exact identifier matching and RRF fusion.

### 20.5 Single vs Iterative Retrieval

Iterative retrieval improved context hit rate when paired with hybrid retrieval, but it did not improve final answer correctness in the current DeepEval run. Likely reasons:

- second-pass query can drift based on partial answer wording;
- generation may become more cautious or more diffuse;
- context precision remains low because top-k still includes many non-target chunks.

The best product choice is single-pass hybrid by default, with iterative exposed as an optional "deep search" mode.

## 21. Tests

Latest full test run:

```text
112 passed, 1 warning
```

Warning:

```text
DeprecationWarning from langchain_community.llms.Ollama in optional RAGAS path
```

Main test files:

| Test file | What it validates |
|---|---|
| `tests/test_ingest.py` | Ingestion filters, raw file records, CLI behavior. |
| `tests/test_chunker.py` | Chunk extraction, docstrings, README chunks, stable IDs, oversized splitting. |
| `tests/test_build_index.py` | Shared chunk corpus for dense and BM25, manifest output. |
| `tests/test_index_dense.py` | Dense index construction, query output, linked chunks, CLI help without heavy dependencies. |
| `tests/test_bm25_index.py` | Tokenization, query matching, persistence, fingerprinting. |
| `tests/test_retriever.py` | RRF scoring, linked chunk preservation, hybrid search calls. |
| `tests/test_generator.py` | Prompt formatting, citations, refusal behavior, identifier guard. |
| `tests/test_pipeline.py` | Single vs iterative behavior, logging, linked chunk deduplication, confidence integration. |
| `tests/test_confidence.py` | Temperature sampling, lexical/semantic scoring, logging, threshold decisions. |
| `tests/test_eval.py` | Test-set validation, metric scoring, refusal scoring, optional RAGAS degradation. |
| `tests/test_ablation.py` | Four-condition ablation artifacts, adversarial scoring, CLI help. |
| `tests/test_deepeval_judge.py` | DeepEval artifact writing, summary averages, expected refusals, mocked judge failures, CLI help. |

The tests use fake retrievers, fake collections, fake generators, and small fixtures so they do not require Ollama or model downloads for normal unit testing.

## 22. Reproducibility Commands

### 22.1 Environment

```bash
cd /home/nicu/facultate/fml/codebase-intelligence-assistant
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

If running optional evaluation:

```bash
python -m pip install -e ".[eval]"
```

### 22.2 Build FastAPI Index

```bash
.venv/bin/python -m src.build_index \
  --repo data/raw/fastapi \
  --include-path-prefix fastapi \
  --include-path-prefix README.md \
  --exclude-tests
```

### 22.3 Validate FastAPI Test Set

```bash
.venv/bin/python -m eval.run_eval --test-set eval/test_set_fastapi.json
```

Expected validation:

```text
Loaded 20 evaluation questions from eval/test_set_fastapi.json
Refusal questions: 1
```

### 22.4 Run FastAPI Ablation

```bash
.venv/bin/python -m eval.ablation \
  --test-set eval/test_set_fastapi.json \
  --json-output results/ablation_fastapi.json \
  --csv-output results/ablation_fastapi.csv \
  --pipeline-log-path results/ablation_fastapi_runs.jsonl \
  --confidence-log-path results/ablation_fastapi_confidence_log.jsonl \
  --no-confidence
```

### 22.5 Run DeepEval Judge

```bash
LOCAL_MODEL_API_KEY=dummy .venv/bin/python -m eval.deepeval_judge \
  --ablation-results results/ablation_fastapi.json \
  --test-set eval/test_set_fastapi.json \
  --json-output results/deepeval_fastapi.json \
  --csv-output results/deepeval_fastapi.csv \
  --model qwen2.5-coder:7b \
  --base-url http://localhost:11434/v1/
```

This command requires Ollama to be running with `qwen2.5-coder:7b` available.

### 22.6 Run Full Tests

```bash
.venv/bin/python -m pytest
```

### 22.7 Run Streamlit UI

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Example UI question:

```text
How does FastAPI apply custom Swagger UI parameters when generating the docs HTML?
```

Expected behavior:

- answer mentions `swagger_ui_parameters`;
- answer cites `get_swagger_ui_html`;
- answer cites `FastAPI.setup.swagger_ui_html` if retrieved;
- retrieved snippets are visible in the expander.

## 23. Current Known Limitations

### 23.1 Language Coverage

The chunker is currently Python-first. Ingestion recognizes JavaScript, TypeScript, and Java files, but semantic chunk extraction is implemented mainly for Python plus README files. Full multi-language support remains future work.

### 23.2 Dense Retrieval Performance

GraphCodeBERT can be slow on CPU. Caching helps, but the first build can take time. Query-time embedding can also try to access HuggingFace metadata if the local cache is incomplete.

### 23.3 Exact Context Matching Is Strict

The evaluation context matcher requires exact filepath and function name matches. Some retrieval results may be semantically close but counted as misses. Example: retrieving `OAuth2PasswordBearer` but not `OAuth2PasswordBearer.__call__` can be counted as a miss for a question targeting `__call__`.

### 23.4 Context Precision Is Low

Even the best condition has context precision around 0.143. This means the expected chunk is usually present, but many extra chunks are also returned. A reranker or tighter top-k selection could improve this.

### 23.5 Confidence Layer Is Untuned

The confidence threshold and similarity method are not tuned enough for production refusal. It over-refuses on the FastAPI QA set.

### 23.6 RAGAS Dependency Stack

RAGAS is optional and currently blocked by a LangChain community dependency issue in this environment. This is not blocking the project because DeepEval GEval is now the main LLM-judge evaluation method.

### 23.7 Generation Quality Depends on Ollama

The generator requires a running Ollama-compatible endpoint and the `qwen2.5-coder:7b` model. If Ollama is unavailable, pipeline generation fails.

## 24. Feature Improvements

### 24.1 Retrieval Improvements

- Add a cross-encoder or LLM reranker after RRF.
- Tune `top_k`, `dense_k_multiplier`, and `linked_window`.
- Add a query classifier for identifier-heavy vs conceptual questions.
- Add file/path filters from UI controls.
- Improve dense embedding text formatting for large classes and nested functions.
- Add symbol graph edges, such as caller/callee, imports, and class inheritance.

### 24.2 Chunking Improvements

- Add full tree-sitter support for JavaScript, TypeScript, and Java.
- Store parent symbol information for methods and nested functions.
- Store imports and references as structured metadata.
- Improve README splitting for long documentation pages.
- Tune oversized chunk splitting by token count instead of line count.

### 24.3 Generation Improvements

- Use a prompt that better distinguishes "partial evidence" from "no evidence".
- Ask the model to produce citations for each bullet and a compact "sources used" list.
- Add a structured output mode with fields: answer, citations, uncertainty, missing evidence.
- Make refusal less brittle by letting the answer say what was found and what was not found.

### 24.4 Confidence and Grounding Improvements

- Tune the confidence threshold on the FastAPI eval set.
- Use confidence as a warning badge instead of an automatic refusal.
- Improve semantic similarity with code-aware embeddings.
- Ground claims against parsed identifiers instead of raw substring search.
- Separate "ungrounded identifier" from "unsupported natural-language claim".

### 24.5 Evaluation Improvements

- Add more DeepEval judge runs after retrieval or prompt changes.
- Keep RAGAS as an optional comparison if the dependency stack is fixed.
- Add more FastAPI questions, especially cross-file and negative questions.
- Add latency metrics by condition to quantify the cost of iterative retrieval.
- Evaluate top-k sensitivity.
- Track exact answer citation correctness, not just answer token overlap.
- Add an oracle-retrieval evaluation to isolate generation quality from retrieval quality.

### 24.6 UI Improvements

- Add a confidence toggle that shows warnings without replacing answers.
- Add source filters by path prefix.
- Add a retrieved snippet score/rank display.
- Add copyable citations.
- Add a "show retrieval trace" expander for debugging.
- Add an index manifest panel so users know which repository is active.

## 25. Recommended Demo Script

1. Start with the problem: codebases are too large to inspect manually, and exact identifiers are often unknown.
2. Show the pipeline diagram: ingestion -> chunking -> dense/BM25 -> RRF -> generation -> UI.
3. Explain why hybrid retrieval matters: dense handles meaning, BM25 handles identifiers.
4. Show the retrieval ablation and DeepEval judge tables:

```text
B hybrid single: best DeepEval answer correctness
D hybrid iterative: best context hit rate
D hybrid iterative: slightly best faithfulness
Dense-only: weak baseline
Adversarial refusal: 100 percent
```

5. Open the Streamlit UI, keep "Fast answer" selected, and ask:

```text
How does FastAPI apply custom Swagger UI parameters when generating the docs HTML?
```

6. Expand retrieved snippets and show cited FastAPI functions.
7. Ask an adversarial question:

```text
Where is the React dashboard build script configured in FastAPI?
```

8. Show that the assistant refuses.
9. Switch to "Deep search" to explain the optional retrieval-coverage mode.
10. Conclude with future work: reranking, confidence warning mode, citation grading, multi-language chunking.

## 26. One-Slide Results Summary

Use these tables in slides.

Retrieval ablation:

| Condition | Retrieval | Passes | Context hit | Answer overlap | Adversarial refusal |
|---|---|---|---:|---:|---:|
| A | Dense | Single | 0.000 | 0.117 | 1.000 |
| B | Hybrid | Single | 0.737 | 0.488 | 1.000 |
| C | Dense | Iterative | 0.053 | 0.105 | 1.000 |
| D | Hybrid | Iterative | 0.895 | 0.362 | 1.000 |

DeepEval judge:

| Condition | Retrieval | Passes | Answer correctness | Faithfulness | Refusal quality |
|---|---|---|---:|---:|---:|
| A | Dense | Single | 0.079 | 0.025 | 0.800 |
| B | Hybrid | Single | 0.700 | 0.260 | 0.800 |
| C | Dense | Iterative | 0.089 | 0.010 | 0.800 |
| D | Hybrid | Iterative | 0.526 | 0.280 | 0.800 |

Main takeaway:

```text
Hybrid retrieval is the key improvement. Hybrid single-pass gives the best answer correctness and is the default UI mode, while hybrid iterative remains useful as optional deep search.
```

## 27. Final Current State

The project currently has:

- working ingestion;
- semantic chunking for Python and README files;
- GraphCodeBERT dense retrieval with Chroma;
- BM25 sparse retrieval;
- RRF hybrid search;
- local Ollama generation;
- single-pass and iterative RAG modes;
- grounding checks;
- optional confidence estimation;
- Streamlit chat UI with Fast answer and Deep search modes;
- manual eval harness;
- FastAPI eval set;
- four-condition ablation runner;
- DeepEval GEval LLM-judge evaluator;
- latest no-confidence FastAPI retrieval results;
- latest DeepEval FastAPI answer-quality results;
- 112 passing tests.

The most important next technical step is to improve answer quality while preserving the strong retrieval gains from hybrid search. The most practical next product step is to add better citation grading and retrieval ranking visibility in the UI.

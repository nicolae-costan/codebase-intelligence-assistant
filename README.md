# Codebase Intelligence Assistant

A retrieval-augmented assistant for understanding software repositories through grounded code search and question answering.

## Overview

This project builds a local-first pipeline that ingests a codebase, splits it into semantic chunks, indexes it with hybrid retrieval, and uses a local code model to answer questions strictly from retrieved context.

The system is designed to:

- understand repository structure and source code semantics
- retrieve both identifier-heavy and conceptually related code
- generate grounded answers with citation-based support
- evaluate retrieval and answer quality with RAG-focused metrics

## Architecture

The pipeline is organized into seven stages:

1. **Ingestion**: crawl a repository and collect source files and documentation.
2. **Semantic Chunking**: split code by functions, classes, and meaningful documentation units.
3. **Embedding and Indexing**: build dense GraphCodeBERT and sparse BM25 indexes, then fuse rankings.
4. **Iterative Retrieval**: use a two-pass RepoCoder-style strategy to refine retrieved context.
5. **Generation**: produce answers with a local model such as `Qwen2.5-Coder 7B` via `Ollama`.
6. **Hallucination Handling**: enforce grounded answers and estimate confidence when needed.
7. **Evaluation**: measure quality with manual checks and RAGAS-style metrics.

## Project Structure

```text
codebase-intelligence-assistant/
  src/            # pipeline modules
  eval/           # evaluation scripts and experiments
  data/           # cloned repositories and cached chunks (gitignored)
    raw/
    chunks/
  index/          # persisted retrieval indexes (gitignored)
    chroma_db/
    bm25/
  results/        # evaluation outputs and tables (gitignored)
  app/            # frontend application
  notebooks/      # exploratory experiments and early prototyping
  docs/           # report, architecture diagram, and submission materials
  tests/          # unit tests and small executable fixtures
```

## Development Setup

Create and activate a virtual environment, then install the project with development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

## Repository Ingestion

T2 implements a lightweight ingestion pipeline for cloning or crawling a repository before chunking. It collects `.py`, `.js`, `.ts`, `.java`, and README files, skips common dependency/build/cache directories, filters binary, license, generated, and oversized files, and can persist the flat raw file list for downstream stages.

Run it on a local repository:

```bash
python3 -m src.ingest --repo tests/fixtures/sample_python_project
```

Clone and ingest a Git repository into `data/raw/<repo-name>`:

```bash
python3 -m src.ingest --url https://github.com/example/project.git
```

Persist a machine-readable raw file list:

```bash
python3 -m src.ingest --repo tests/fixtures/sample_python_project --output data/raw/files.json
```

Emit raw file records to stdout:

```bash
python3 -m src.ingest --repo tests/fixtures/sample_python_project --json
```

## Semantic Chunker

T1 implements a Python-first semantic chunker with README support. It extracts functions, async functions, classes, methods, top-level module imports/constants as `__module__`, and README text chunks, preserving source text, docstrings, file paths, and line ranges for downstream retrieval and citation.

Run it on a repository or directory:

```bash
python -m src.chunker --repo tests/fixtures/sample_python_project
```

Emit machine-readable chunks for indexing:

```bash
python -m src.chunker --repo tests/fixtures/sample_python_project --json
```

Each chunk includes:

- `id`
- `filepath`
- `function_name`
- `start_line`
- `end_line`
- `docstring`
- `source`
- `language`

## Dense Retrieval

T4 implements GraphCodeBERT dense retrieval with ChromaDB persistence. It embeds semantic chunks with `microsoft/graphcodebert-base`, caches vectors in `data/chunks/embeddings.pkl`, and stores a queryable Chroma index in `index/chroma_db/`.

For normal use, build both dense and BM25 indexes from one shared chunk corpus:

```bash
python -m src.build_index --repo tests/fixtures/sample_python_project
```

The combined builder writes `index/manifest.json` with the repository path, chunk count, shared chunk fingerprint, dense model, Chroma collection, and BM25 path. This is the preferred indexing path because hybrid retrieval assumes dense and sparse indexes were built from the same chunks.

Build only a dense index for focused debugging:

```bash
python -m src.index_dense --repo tests/fixtures/sample_python_project
```

Build and query in one command:

```bash
python -m src.index_dense --repo tests/fixtures/sample_python_project --query "where is greeting built" --top-k 3
```

For large repositories, filter the indexed paths when you want implementation code rather than tests or docs:

```bash
python -m src.index_dense \
  --repo data/raw/fastapi \
  --include-path-prefix fastapi \
  --exclude-tests \
  --query "where is request validation handled?" \
  --top-k 5
```

The same filtering flags are available on the combined builder:

```bash
python -m src.build_index \
  --repo data/raw/fastapi \
  --include-path-prefix fastapi \
  --exclude-tests
```

Oversized functions and classes are split into overlapping linked subchunks by default during dense indexing. When a subchunk matches, `--linked-window` includes neighboring parts from the same symbol in the query output:

```bash
python -m src.index_dense \
  --repo data/raw/fastapi \
  --include-path-prefix fastapi \
  --max-chunk-lines 220 \
  --chunk-overlap-lines 20 \
  --linked-window 1 \
  --query "where is request validation handled?" \
  --top-k 5
```

Use it from code:

```python
from src.chunker import chunk_repository
from src.index_dense import index_chunks, query_dense

chunks = chunk_repository("tests/fixtures/sample_python_project")
index_chunks(chunks)
results = query_dense("where is the greeting assembled?", k=3)
```

The first run downloads the HuggingFace model and can be slow on CPU. Subsequent runs reuse the embedding cache unless chunk content, chunk ids, or model name change.

## Generation

T8 implements a standalone Ollama generation wrapper for hardcoded or retrieved chunks. Install and run the local model with Ollama:

```bash
ollama pull qwen2.5-coder:7b
```

By default, generation uses `http://localhost:11434/v1`. Override it when Ollama runs elsewhere:

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
```

The API key defaults to `ollama`, or you can set `OLLAMA_API_KEY` if your endpoint requires one.

Run a hardcoded-context smoke test:

```bash
python -m src.generator --fixture --query "What does Greeter.greet do?"
```

Use it from code:

```python
from src.generator import generate

answer = generate(context_chunks=[chunk], query="What does this function do?")
```

## HonestCoder Confidence

T10 adds an opt-in confidence layer on top of the final retrieved context. It samples three answers at temperatures `0.0`, `0.5`, and `0.9`, combines lexical agreement with MiniLM semantic similarity, and logs the confidence decision to `results/confidence_log.jsonl`.

Run the pipeline with confidence enabled:

```bash
python -m src.pipeline \
  --query "What fields are stored on Chunk?" \
  --mode iterative \
  --confidence \
  --json
```

Tune the low-confidence cutoff or disable confidence logging:

```bash
python -m src.pipeline \
  --query "What fields are stored on Chunk?" \
  --confidence \
  --confidence-threshold 0.75 \
  --confidence-log-path none
```

## Evaluation Harness

T3 adds a manual question-answer evaluation set and a small harness that can score any pipeline function shaped like `pipeline_fn(question) -> answer_or_result`. The committed test set lives in `eval/test_set.json` and includes answerable questions, expected source contexts, and out-of-scope refusal cases.

Validate the test set:

```bash
python3 -m eval.run_eval
```

Smoke-test the harness with oracle answers:

```bash
python3 -m eval.run_eval --oracle
```

Use the harness from code:

```python
from eval.run_eval import run_eval

metrics = run_eval(pipeline_fn)
```

Optional RAGAS scoring requires the eval extra and a local Ollama-compatible judge model:

```bash
python3 -m pip install -e ".[eval]"
RAGAS_OLLAMA_MODEL=qwen2.5-coder:7b python3 -m eval.run_eval --oracle --ragas
```

## Notes

- `data/`, `index/`, and `results/` are intentionally ignored by git.
- `notebooks/` is available for fast experimentation, but reusable code should move into `src/`.
- `docs/` is reserved for final reports, diagrams, and short design notes for non-obvious decisions.

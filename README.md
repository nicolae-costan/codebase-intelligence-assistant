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

T1 implements a Python-first semantic chunker. It extracts functions, async functions, classes, and methods from `.py` files, preserving source text, docstrings, file paths, and line ranges for downstream retrieval and citation.

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

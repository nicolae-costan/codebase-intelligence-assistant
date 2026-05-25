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

## Notes

- `data/`, `index/`, and `results/` are intentionally ignored by git.
- `notebooks/` is available for fast experimentation, but reusable code should move into `src/`.
- `docs/` is reserved for final reports, diagrams, and short design notes for non-obvious decisions.

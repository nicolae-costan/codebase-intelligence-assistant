# Codebase Intelligence Assistant

A retrieval-augmented assistant for understanding software repositories through grounded code search and question answering.

## Overview

This project builds a pipeline that ingests a codebase, splits it into semantic chunks, indexes it with hybrid retrieval, and uses a local code model to answer questions strictly from retrieved context.

The system is designed to:
- understand repository structure and source code semantics
- retrieve both identifier-heavy and conceptually related code
- generate grounded answers with citation-based support
- evaluate retrieval and answer quality with RAG-focused metrics

## Architecture

The pipeline is organized into seven stages:

1. **Ingestion**  
   Crawl a repository and collect source files and documentation.

2. **Semantic Chunking**  
   Split code by functions, classes, and meaningful documentation units.

3. **Embedding and Indexing**  
   Build a hybrid retrieval layer using:
   - dense embeddings with `GraphCodeBERT`
   - sparse retrieval with `BM25`
   - fused ranking with Reciprocal Rank Fusion

4. **Iterative Retrieval**  
   Use a two-pass retrieval strategy inspired by RepoCoder to refine context before final generation.

5. **Generation**  
   Produce answers with a local small language model, currently planned around `Qwen2.5-Coder 7B` via `Ollama`.

6. **Hallucination Handling**  
   Enforce grounded answers and estimate confidence through multi-sample consistency checks.

7. **Evaluation**  
   Measure quality with `RAGAS`, including faithfulness, context precision, context recall, and answer relevance.

## Project Structure

```text
codebase-intelligence-assistant/
├── src/            # pipeline modules
├── eval/           # evaluation scripts and experiments
├── data/           # cloned repositories and cached chunks (gitignored)
│   ├── raw/
│   └── chunks/
├── index/          # persisted retrieval indexes (gitignored)
│   ├── chroma_db/
│   └── bm25/
├── results/        # evaluation outputs and tables (gitignored)
├── app/            # frontend application
├── notebooks/      # exploratory experiments and early prototyping
└── docs/           # report, architecture diagram, and submission materials
```

## Tech Stack

- **Language:** Python
- **Parsing:** `pathlib`, `gitpython`, `tree-sitter`
- **Dense retrieval:** `GraphCodeBERT`
- **Sparse retrieval:** `rank_bm25`
- **Vector store:** `ChromaDB`
- **Model runtime:** `Ollama`
- **Evaluation:** `RAGAS`
- **Frontend:** Streamlit or React

## Current Scope

The initial implementation focuses on:
- repository ingestion
- AST-aware chunking
- hybrid retrieval setup
- iterative retrieval experiments
- grounded code question answering
- evaluation and ablation support

## Notes

- `data/`, `index/`, and `results/` are intentionally ignored by git.
- `notebooks/` is available for fast experimentation, but reusable code should move into `src/`.
- `docs/` is reserved for the final report and architecture artifacts.

"""Canonical dense + BM25 index builder for one shared chunk corpus."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.bm25_index import DEFAULT_INDEX_PATH as DEFAULT_BM25_PATH
from src.bm25_index import build_and_save, fingerprint_chunks
from src.chunker import DEFAULT_CHUNK_OVERLAP_LINES, chunk_repository
from src.index_dense import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_PATH,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MAX_CHUNK_LINES,
    MODEL_NAME,
    filter_chunks_by_path,
    index_chunks,
)

DEFAULT_MANIFEST_PATH = Path("index/manifest.json")


def build_indexes(
    repo: str | Path,
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    chroma_path: str | Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    bm25_path: str | Path = DEFAULT_BM25_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_chunk_lines: int | None = DEFAULT_MAX_CHUNK_LINES,
    chunk_overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
    include_path_prefixes: Sequence[str] = (),
    exclude_path_prefixes: Sequence[str] = (),
    exclude_tests: bool = False,
    force_rebuild: bool = False,
    embedder: Any | None = None,
    collection: Any | None = None,
    device: str | None = None,
    show_progress: bool = False,
) -> dict[str, object]:
    """Build dense and BM25 indexes from the same filtered chunk list."""

    repo_path = Path(repo).resolve()
    chunks = chunk_repository(
        repo_path,
        max_chunk_lines=max_chunk_lines,
        chunk_overlap_lines=chunk_overlap_lines,
    )
    chunks = filter_chunks_by_path(
        chunks,
        include_path_prefixes=include_path_prefixes,
        exclude_path_prefixes=exclude_path_prefixes,
        exclude_tests=exclude_tests,
    )
    chunk_fingerprint = fingerprint_chunks(chunks)

    dense_index = index_chunks(
        chunks,
        cache_path=cache_path,
        persist_dir=chroma_path,
        collection_name=collection_name,
        model_name=model_name,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
        embedder=embedder,
        collection=collection,
        device=device,
        show_progress=show_progress,
    )
    bm25_index = build_and_save(chunks, bm25_path)
    manifest = _manifest(
        repo_path=repo_path,
        chunks_count=len(chunks),
        chunk_fingerprint=chunk_fingerprint,
        model_name=model_name,
        collection_name=collection_name,
        chroma_path=chroma_path,
        bm25_path=bm25_path,
        cache_path=cache_path,
        include_path_prefixes=include_path_prefixes,
        exclude_path_prefixes=exclude_path_prefixes,
        exclude_tests=exclude_tests,
        max_chunk_lines=max_chunk_lines,
        chunk_overlap_lines=chunk_overlap_lines,
    )
    write_manifest(manifest, manifest_path)

    return {
        "chunks": chunks,
        "fingerprint": chunk_fingerprint,
        "dense_index": dense_index,
        "bm25_index": bm25_index,
        "manifest": manifest,
    }


def write_manifest(manifest: dict[str, object], path: str | Path = DEFAULT_MANIFEST_PATH) -> Path:
    """Persist a JSON manifest describing the indexed corpus."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    """Build both retrieval indexes for a repository."""

    parser = argparse.ArgumentParser(description="Build dense Chroma and BM25 indexes from one chunk corpus.")
    parser.add_argument("--repo", required=True, help="Repository or directory path to chunk and index.")
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH), help="Embedding cache path.")
    parser.add_argument("--index-path", default=str(DEFAULT_CHROMA_PATH), help="Chroma persistence directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--bm25-index-path", default=str(DEFAULT_BM25_PATH), help="BM25 pickle path.")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH), help="JSON manifest path.")
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument(
        "--max-chunk-lines",
        type=int,
        default=DEFAULT_MAX_CHUNK_LINES,
        help="Split symbols longer than this many lines into overlapping linked subchunks. Use 0 to disable.",
    )
    parser.add_argument(
        "--chunk-overlap-lines",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_LINES,
        help="Line overlap between subchunks created by --max-chunk-lines.",
    )
    parser.add_argument(
        "--include-path-prefix",
        action="append",
        default=[],
        help="Only index chunks whose repository-relative path starts with this prefix. Repeatable.",
    )
    parser.add_argument(
        "--exclude-path-prefix",
        action="append",
        default=[],
        help="Skip chunks whose repository-relative path starts with this prefix. Repeatable.",
    )
    parser.add_argument("--exclude-tests", action="store_true", help="Skip common test directories and test_*.py files.")
    parser.add_argument("--device", help="Torch device to use, such as 'cuda', 'cuda:0', or 'cpu'.")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore any existing embedding cache.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging.")
    args = parser.parse_args(argv)

    max_chunk_lines = args.max_chunk_lines if args.max_chunk_lines > 0 else None
    result = build_indexes(
        args.repo,
        cache_path=args.cache_path,
        chroma_path=args.index_path,
        collection_name=args.collection,
        bm25_path=args.bm25_index_path,
        manifest_path=args.manifest_path,
        model_name=args.model,
        batch_size=args.batch_size,
        max_chunk_lines=max_chunk_lines,
        chunk_overlap_lines=args.chunk_overlap_lines,
        include_path_prefixes=args.include_path_prefix,
        exclude_path_prefixes=args.exclude_path_prefix,
        exclude_tests=args.exclude_tests,
        force_rebuild=args.force_rebuild,
        device=args.device,
        show_progress=not args.quiet,
    )
    manifest = result["manifest"]
    print(f"Indexed {manifest['chunk_count']} chunks")
    print(f"Fingerprint: {manifest['chunk_fingerprint']}")
    print(f"Chroma index: {args.index_path}")
    print(f"BM25 index: {args.bm25_index_path}")
    print(f"Manifest: {args.manifest_path}")
    return 0


def _manifest(
    *,
    repo_path: Path,
    chunks_count: int,
    chunk_fingerprint: str,
    model_name: str,
    collection_name: str,
    chroma_path: str | Path,
    bm25_path: str | Path,
    cache_path: str | Path,
    include_path_prefixes: Sequence[str],
    exclude_path_prefixes: Sequence[str],
    exclude_tests: bool,
    max_chunk_lines: int | None,
    chunk_overlap_lines: int,
) -> dict[str, object]:
    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "repo_path": str(repo_path),
        "chunk_count": chunks_count,
        "chunk_fingerprint": chunk_fingerprint,
        "dense_model": model_name,
        "chroma_collection": collection_name,
        "chroma_path": str(chroma_path),
        "bm25_path": str(bm25_path),
        "embedding_cache_path": str(cache_path),
        "include_path_prefixes": list(include_path_prefixes),
        "exclude_path_prefixes": list(exclude_path_prefixes),
        "exclude_tests": exclude_tests,
        "max_chunk_lines": max_chunk_lines,
        "chunk_overlap_lines": chunk_overlap_lines,
    }


if __name__ == "__main__":
    raise SystemExit(main())

"""BM25 sparse index for semantic chunk retrieval.

Builds a BM25 index over :class:`~src.schema.Chunk` objects using the
``rank_bm25`` library.  Tokenization is camelCase- and snake_case-aware so
that identifiers like ``processPayment`` score correctly when a user searches
for "process payment".

The index is persisted to disk with :mod:`pickle` and reloaded automatically
on startup when the index file already exists.

Typical usage::

    from src.bm25_index import BM25Index
    from src.chunker import chunk_repository

    chunks = chunk_repository("/path/to/repo")
    idx = BM25Index.build(chunks)
    idx.save()                         # → index/bm25/bm25.pkl

    # Later / in a new process:
    idx = BM25Index.load()
    results = idx.query_bm25("process payment", k=5)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Sequence

from src.chunker import chunk_repository
from src.schema import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default persistence path
# ---------------------------------------------------------------------------

DEFAULT_INDEX_PATH = Path("index/bm25/bm25.pkl")
_PERSISTENCE_TYPE = "src.bm25_index.BM25Index"
_PERSISTENCE_VERSION = 1

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_ACRONYM_BOUNDARY_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_CASE_RE = re.compile(r"([a-z0-9])([A-Z])")
_NON_WORD_RE = re.compile(r"[^a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split *text* into lowercase tokens, handling camelCase and snake_case.

    Steps:
    1. Insert spaces at acronym and camelCase boundaries, e.g.
       ``HTTPSConnection`` → ``HTTPS Connection`` and ``processPayment`` →
       ``process Payment``.
    2. Split on any run of non-alphanumeric characters (covers ``_``, ``-``,
       whitespace, punctuation, …).
    3. Lowercase and discard empty tokens.

    Examples::

        >>> tokenize("processPayment")
        ['process', 'payment']
        >>> tokenize("get_user_by_id")
        ['get', 'user', 'by', 'id']
        >>> tokenize("HTTPSConnection")
        ['https', 'connection']
    """
    expanded = _ACRONYM_BOUNDARY_RE.sub(r"\1 \2", text)
    expanded = _CAMEL_CASE_RE.sub(r"\1 \2", expanded)
    # split on non-word chars (handles snake_case, kebab-case, whitespace…)
    raw_tokens = _NON_WORD_RE.split(expanded)
    return [t.lower() for t in raw_tokens if t]


def _chunk_document(chunk: Chunk) -> str:
    """Concatenate the fields used for BM25 scoring."""
    parts = [
        chunk.function_name,
        chunk.docstring,
        chunk.source,
        chunk.filepath,
    ]
    return " ".join(filter(None, parts))


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


class BM25Index:
    """A BM25 index over a collection of :class:`~src.schema.Chunk` objects.

    Attributes:
        chunks: The ordered list of chunks that were indexed.
    """

    def __init__(self, chunks: list[Chunk], _bm25: object, fingerprint: str | None = None) -> None:
        self._chunks = chunks
        self._bm25 = _bm25
        self._fingerprint = fingerprint or fingerprint_chunks(chunks)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, chunks: Sequence[Chunk]) -> "BM25Index":
        """Build a BM25 index from *chunks*.

        Args:
            chunks: Iterable of :class:`~src.schema.Chunk` objects to index.

        Returns:
            A fully constructed :class:`BM25Index` ready for querying.

        Raises:
            ImportError: If ``rank_bm25`` is not installed.
            ValueError: If *chunks* is empty.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rank_bm25 is required for BM25 indexing. "
                "Install it with: pip install rank-bm25"
            ) from exc

        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("Cannot build a BM25 index from an empty chunk list.")

        corpus = [tokenize(_chunk_document(c)) for c in chunk_list]
        bm25 = BM25Okapi(corpus)
        logger.info("Built BM25 index over %d chunks.", len(chunk_list))
        return cls(chunk_list, bm25, fingerprint=fingerprint_chunks(chunk_list))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_bm25(self, query: str, k: int = 10) -> list[Chunk]:
        """Return the top-*k* chunks most relevant to *query*.

        The query is tokenized with the same :func:`tokenize` function used at
        index time, so camelCase/snake_case search terms are handled correctly.

        Args:
            query: Free-text or identifier query string.
            k: Maximum number of results to return.

        Returns:
            Up to *k* :class:`~src.schema.Chunk` objects ranked by BM25 score,
            highest score first.  Chunks with a score of 0 are excluded.
        """
        if not query.strip():
            return []

        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair each chunk with its score, filter zeros, sort descending
        ranked = sorted(
            ((score, i) for i, score in enumerate(scores) if score > 0),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [self._chunks[i] for _, i in ranked[:k]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = DEFAULT_INDEX_PATH) -> Path:
        """Persist the index to *path* using :mod:`pickle`.

        The parent directory is created automatically.

        Args:
            path: Destination file path (default: ``index/bm25/bm25.pkl``).

        Returns:
            The resolved :class:`~pathlib.Path` where the index was written.
        """
        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            pickle.dump(_index_payload(self), fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved to %s.", dest)
        return dest

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INDEX_PATH) -> "BM25Index":
        """Load a previously saved index from *path*.

        Args:
            path: Source file path (default: ``index/bm25/bm25.pkl``).

        Returns:
            A :class:`BM25Index` deserialized from the file.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        src_path = Path(path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {src_path}")
        with src_path.open("rb") as fh:
            payload = _BM25Unpickler(fh).load()
        index = _index_from_payload(payload, src_path)
        logger.info("BM25 index loaded from %s (%d chunks).", src_path, len(index._chunks))
        return index

    @staticmethod
    def exists(path: str | Path = DEFAULT_INDEX_PATH) -> bool:
        """Return *True* if a persisted index file exists at *path*."""
        return Path(path).resolve().exists()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def chunks(self) -> list[Chunk]:
        """The ordered list of chunks that were indexed."""
        return list(self._chunks)

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint of the chunks used to build the index."""
        return self._fingerprint

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"BM25Index(chunks={len(self._chunks)})"


class _BM25Unpickler(pickle.Unpickler):
    """Load current payloads and legacy ``python -m`` BM25Index pickles."""

    def find_class(self, module: str, name: str) -> object:
        if name == "BM25Index" and module in {"__main__", "src.pipeline"}:
            return BM25Index
        return super().find_class(module, name)


def _index_payload(index: BM25Index) -> dict[str, object]:
    return {
        "type": _PERSISTENCE_TYPE,
        "version": _PERSISTENCE_VERSION,
        "chunks": index._chunks,
        "bm25": index._bm25,
        "fingerprint": index._fingerprint,
    }


def _index_from_payload(payload: object, src_path: Path) -> BM25Index:
    if isinstance(payload, BM25Index):
        index = payload
        if not hasattr(index, "_fingerprint"):
            index._fingerprint = fingerprint_chunks(index._chunks)
        return index

    if isinstance(payload, dict) and payload.get("type") == _PERSISTENCE_TYPE:
        if payload.get("version") != _PERSISTENCE_VERSION:
            raise TypeError(f"Unsupported BM25 index version in pickle at {src_path}.")
        chunks = payload.get("chunks")
        bm25 = payload.get("bm25")
        fingerprint = payload.get("fingerprint")
        if not isinstance(chunks, list) or bm25 is None:
            raise TypeError(f"Pickle at {src_path} did not contain a valid BM25Index payload.")
        return BM25Index(
            chunks=chunks,
            _bm25=bm25,
            fingerprint=str(fingerprint) if fingerprint else fingerprint_chunks(chunks),
        )

    raise TypeError(f"Pickle at {src_path} did not contain a BM25Index.")


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------


def build_and_save(
    chunks: Sequence[Chunk],
    path: str | Path = DEFAULT_INDEX_PATH,
) -> BM25Index:
    """Build a :class:`BM25Index` from *chunks* and persist it to *path*.

    This is a convenience wrapper around :meth:`BM25Index.build` +
    :meth:`BM25Index.save`.

    Args:
        chunks: Chunks to index.
        path: Destination pickle file (default: ``index/bm25/bm25.pkl``).

    Returns:
        The newly built and saved :class:`BM25Index`.
    """
    index = BM25Index.build(chunks)
    index.save(path)
    return index


def fingerprint_chunks(chunks: Sequence[Chunk]) -> str:
    """Return a stable fingerprint for a chunk sequence."""

    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.filepath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.function_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.start_line).encode("ascii"))
        digest.update(b":")
        digest.update(str(chunk.end_line).encode("ascii"))
        digest.update(b"\0")
        digest.update(_chunk_document(chunk).encode("utf-8"))
        digest.update(b"\0\0")
    return digest.hexdigest()


def load_or_build(
    chunks: Sequence[Chunk],
    path: str | Path = DEFAULT_INDEX_PATH,
    *,
    force_rebuild: bool = False,
) -> BM25Index:
    """Load the index from disk if it exists, otherwise build and save it.

    This is the recommended startup pattern: call this once with all chunks
    and the function will reuse a cached index whenever the persisted index
    fingerprint matches the current chunks.

    Args:
        chunks: Chunks to use if the index must be built from scratch.
        path: Pickle file path (default: ``index/bm25/bm25.pkl``).
        force_rebuild: Rebuild the index even if a matching pickle exists.

    Returns:
        A ready-to-query :class:`BM25Index`.
    """
    chunk_list = list(chunks)
    expected_fingerprint = fingerprint_chunks(chunk_list)
    if BM25Index.exists(path) and not force_rebuild:
        logger.info("Found existing BM25 index at %s, loading from disk.", path)
        index = BM25Index.load(path)
        if index.fingerprint == expected_fingerprint:
            return index
        logger.info("Existing BM25 index is stale; rebuilding %s.", path)
    logger.info("No BM25 index found at %s, building from scratch.", path)
    return build_and_save(chunk_list, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Build or query the BM25 sparse index from the command line."""

    parser = argparse.ArgumentParser(description="Build and query a BM25 sparse index.")
    parser.add_argument("--repo", required=True, help="Repository or directory path to chunk and index.")
    parser.add_argument("--query", help="Optional query to run after indexing.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of BM25 results to return for --query.")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="BM25 pickle path.")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore any existing BM25 pickle.")
    parser.add_argument("--json", action="store_true", help="Emit query results as JSON.")
    args = parser.parse_args(argv)

    chunks = chunk_repository(args.repo)
    index = load_or_build(chunks, args.index_path, force_rebuild=args.force_rebuild)

    if not args.json:
        print(f"Indexed {len(index)} chunks")
        print(f"BM25 index: {Path(args.index_path)}")

    if args.query:
        results = index.query_bm25(args.query, k=args.top_k)
        if args.json:
            print(json.dumps([chunk.to_dict() for chunk in results], indent=2))
        else:
            print()
            print(f"Top {len(results)} BM25 results")
            for rank, chunk in enumerate(results, start=1):
                print(
                    f"{rank}. {chunk.filepath}::{chunk.function_name} "
                    f"lines {chunk.start_line}-{chunk.end_line}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

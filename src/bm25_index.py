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

import logging
import pickle
import re
from pathlib import Path
from typing import Sequence

from src.schema import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default persistence path
# ---------------------------------------------------------------------------

DEFAULT_INDEX_PATH = Path("index/bm25/bm25.pkl")

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_CAMEL_CASE_RE = re.compile(r"([a-z])([A-Z])")
_NON_WORD_RE = re.compile(r"[^a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split *text* into lowercase tokens, handling camelCase and snake_case.

    Steps:
    1. Insert a space before every uppercase letter that follows a lowercase
       letter (e.g. ``processPayment`` → ``process Payment``).
    2. Split on any run of non-alphanumeric characters (covers ``_``, ``-``,
       whitespace, punctuation, …).
    3. Lowercase and discard empty tokens.

    Examples::

        >>> tokenize("processPayment")
        ['process', 'payment']
        >>> tokenize("get_user_by_id")
        ['get', 'user', 'by', 'id']
        >>> tokenize("HTTPSConnection")
        ['h', 't', 't', 'p', 's', 'connection']
    """
    # camelCase split
    expanded = _CAMEL_CASE_RE.sub(r"\1 \2", text)
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

    def __init__(self, chunks: list[Chunk], _bm25: object) -> None:
        self._chunks = chunks
        self._bm25 = _bm25

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
        return cls(chunk_list, bm25)

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
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
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
            index = pickle.load(fh)
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

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"BM25Index(chunks={len(self._chunks)})"


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


def load_or_build(
    chunks: Sequence[Chunk],
    path: str | Path = DEFAULT_INDEX_PATH,
) -> BM25Index:
    """Load the index from disk if it exists, otherwise build and save it.

    This is the recommended startup pattern: call this once with all chunks
    and the function will reuse a cached index whenever possible.

    Args:
        chunks: Chunks to use if the index must be built from scratch.
        path: Pickle file path (default: ``index/bm25/bm25.pkl``).

    Returns:
        A ready-to-query :class:`BM25Index`.
    """
    if BM25Index.exists(path):
        logger.info("Found existing BM25 index at %s, loading from disk.", path)
        return BM25Index.load(path)
    logger.info("No BM25 index found at %s, building from scratch.", path)
    return build_and_save(chunks, path)

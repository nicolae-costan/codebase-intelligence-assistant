"""Tests for src.bm25_index — BM25 sparse index."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src.bm25_index import (
    BM25Index,
    DEFAULT_INDEX_PATH,
    build_and_save,
    load_or_build,
    tokenize,
)
from src.schema import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    function_name: str,
    source: str = "",
    docstring: str = "",
    filepath: str = "src/module.py",
    chunk_id: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id or function_name,
        filepath=filepath,
        function_name=function_name,
        start_line=1,
        end_line=10,
        docstring=docstring,
        source=source,
        language="python",
    )


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_plain_words(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_camel_case(self):
        assert tokenize("processPayment") == ["process", "payment"]

    def test_snake_case(self):
        assert tokenize("get_user_by_id") == ["get", "user", "by", "id"]

    def test_mixed_camel_and_snake(self):
        tokens = tokenize("processPayment_amount")
        assert "process" in tokens
        assert "payment" in tokens
        assert "amount" in tokens

    def test_lowercases_output(self):
        tokens = tokenize("MyClass")
        assert all(t == t.lower() for t in tokens)

    def test_empty_string(self):
        assert tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = tokenize("foo.bar(baz)")
        assert "foo" in tokens
        assert "bar" in tokens
        assert "baz" in tokens

    def test_numbers_kept(self):
        tokens = tokenize("version2")
        assert "version2" in tokens

    def test_multiple_spaces(self):
        assert tokenize("a   b") == ["a", "b"]


# ---------------------------------------------------------------------------
# BM25Index.build
# ---------------------------------------------------------------------------


class TestBuild:
    def test_basic_build(self):
        chunks = [_make_chunk("processPayment"), _make_chunk("getUserById")]
        idx = BM25Index.build(chunks)
        assert len(idx) == 2

    def test_empty_chunks_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BM25Index.build([])

    def test_repr(self):
        idx = BM25Index.build([_make_chunk("foo")])
        assert "BM25Index" in repr(idx)
        assert "1" in repr(idx)

    def test_chunks_property_returns_copy(self):
        chunks = [_make_chunk("a"), _make_chunk("b")]
        idx = BM25Index.build(chunks)
        # modifying the returned list must not affect the internal state
        result = idx.chunks
        result.clear()
        assert len(idx) == 2


# ---------------------------------------------------------------------------
# BM25Index.query_bm25
# ---------------------------------------------------------------------------


class TestQueryBM25:
    @pytest.fixture()
    def index(self) -> BM25Index:
        chunks = [
            _make_chunk("processPayment", source="def processPayment(amount): pass"),
            _make_chunk("getUserById", source="def getUserById(user_id): pass"),
            _make_chunk("validateEmail", source="def validateEmail(email): pass"),
            _make_chunk("send_invoice", source="def send_invoice(invoice_id): pass"),
        ]
        return BM25Index.build(chunks)

    def test_returns_most_relevant_chunk(self, index: BM25Index):
        results = index.query_bm25("process payment", k=1)
        assert len(results) == 1
        assert results[0].function_name == "processPayment"

    def test_snake_case_query(self, index: BM25Index):
        results = index.query_bm25("get_user", k=1)
        assert results[0].function_name == "getUserById"

    def test_k_limits_results(self, index: BM25Index):
        results = index.query_bm25("def", k=2)
        assert len(results) <= 2

    def test_empty_query_returns_empty(self, index: BM25Index):
        assert index.query_bm25("", k=5) == []

    def test_whitespace_only_query_returns_empty(self, index: BM25Index):
        assert index.query_bm25("   ", k=5) == []

    def test_zero_score_chunks_excluded(self, index: BM25Index):
        # "xyzzy" is not in any chunk — results should be empty, not four 0-scored chunks
        results = index.query_bm25("xyzzy", k=10)
        assert results == []

    def test_result_type(self, index: BM25Index):
        results = index.query_bm25("email validate", k=1)
        assert all(isinstance(c, Chunk) for c in results)

    def test_camel_case_query_matches_correctly(self, index: BM25Index):
        # Query with camelCase term should still resolve correctly
        results = index.query_bm25("sendInvoice", k=1)
        assert results[0].function_name == "send_invoice"


# ---------------------------------------------------------------------------
# Persistence — save / load
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_file(self, tmp_path: Path):
        chunks = [_make_chunk("foo"), _make_chunk("bar")]
        idx = BM25Index.build(chunks)
        dest = tmp_path / "bm25.pkl"
        idx.save(dest)
        assert dest.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        chunks = [_make_chunk("foo")]
        idx = BM25Index.build(chunks)
        dest = tmp_path / "a" / "b" / "c" / "bm25.pkl"
        idx.save(dest)
        assert dest.exists()

    def test_load_round_trip(self, tmp_path: Path):
        chunks = [
            _make_chunk("alpha", source="def alpha(): pass"),
            _make_chunk("beta", source="def beta(): return 42"),
        ]
        idx = BM25Index.build(chunks)
        dest = tmp_path / "bm25.pkl"
        idx.save(dest)

        loaded = BM25Index.load(dest)
        assert len(loaded) == 2
        assert {c.function_name for c in loaded.chunks} == {"alpha", "beta"}

    def test_load_preserves_query_results(self, tmp_path: Path):
        # Need ≥ 4 chunks: BM25Okapi IDF = log((N-f+0.5)/(f+0.5)).
        # With N=2 and f=1 that equals log(1)=0, making all scores zero.
        # At N=4 and f=1 the IDF is positive and ranking works correctly.
        chunks = [
            _make_chunk("processPayment", source="def processPayment(amount): pass"),
            _make_chunk("getUserById", source="def getUserById(user_id): pass"),
            _make_chunk("validateEmail", source="def validateEmail(email): pass"),
            _make_chunk("sendInvoice", source="def sendInvoice(invoice_id): pass"),
        ]
        idx = BM25Index.build(chunks)
        dest = tmp_path / "bm25.pkl"
        idx.save(dest)

        loaded = BM25Index.load(dest)
        results = loaded.query_bm25("process payment", k=1)
        assert len(results) == 1
        assert results[0].function_name == "processPayment"

    def test_load_missing_file_raises(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError):
            BM25Index.load(missing)

    def test_exists_true(self, tmp_path: Path):
        chunks = [_make_chunk("foo")]
        idx = BM25Index.build(chunks)
        dest = tmp_path / "bm25.pkl"
        idx.save(dest)
        assert BM25Index.exists(dest) is True

    def test_exists_false(self, tmp_path: Path):
        assert BM25Index.exists(tmp_path / "missing.pkl") is False


# ---------------------------------------------------------------------------
# Module-level helpers: build_and_save / load_or_build
# ---------------------------------------------------------------------------


class TestConvenienceHelpers:
    def test_build_and_save(self, tmp_path: Path):
        chunks = [_make_chunk("foo"), _make_chunk("bar")]
        dest = tmp_path / "bm25.pkl"
        idx = build_and_save(chunks, dest)
        assert isinstance(idx, BM25Index)
        assert dest.exists()

    def test_load_or_build_builds_when_missing(self, tmp_path: Path):
        chunks = [_make_chunk("foo")]
        dest = tmp_path / "bm25.pkl"
        idx = load_or_build(chunks, dest)
        assert isinstance(idx, BM25Index)
        assert dest.exists()

    def test_load_or_build_loads_when_present(self, tmp_path: Path):
        chunks = [_make_chunk("foo")]
        dest = tmp_path / "bm25.pkl"
        # Build and save first
        original = build_and_save(chunks, dest)
        mtime_before = dest.stat().st_mtime

        # Calling load_or_build again must not rebuild (same mtime)
        loaded = load_or_build(chunks, dest)
        assert dest.stat().st_mtime == mtime_before
        assert len(loaded) == len(original)

import subprocess
import sys
from pathlib import Path
from typing import Sequence

from src.index_dense import (
    DenseIndex,
    chunk_to_embedding_text,
    filter_chunks_by_path,
    index_chunks,
    load_or_compute_embeddings,
)
from src.schema import Chunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, object]],
    ) -> None:
        for chunk_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.records[chunk_id] = {
                "embedding": list(embedding),
                "document": document,
                "metadata": dict(metadata),
            }

    def query(self, query_embeddings: Sequence[Sequence[float]], n_results: int, include: Sequence[str]) -> dict[str, object]:
        query_embedding = list(query_embeddings[0])
        ranked = sorted(
            self.records.items(),
            key=lambda item: _dot(query_embedding, item[1]["embedding"]),
            reverse=True,
        )[:n_results]
        return {
            "ids": [[chunk_id for chunk_id, _ in ranked]],
            "documents": [[record["document"] for _, record in ranked]],
            "metadatas": [[record["metadata"] for _, record in ranked]],
            "distances": [[1.0 - _dot(query_embedding, record["embedding"]) for _, record in ranked]],
        }

    def get(self, where: dict[str, object], include: Sequence[str]) -> dict[str, object]:
        matched = [
            (chunk_id, record)
            for chunk_id, record in self.records.items()
            if all(record["metadata"].get(key) == value for key, value in where.items())
        ]
        return {
            "ids": [chunk_id for chunk_id, _ in matched],
            "documents": [record["document"] for _, record in matched],
            "metadatas": [record["metadata"] for _, record in matched],
        }


def test_chunk_to_embedding_text_includes_retrieval_fields() -> None:
    chunk = _chunk("alpha", "src/app.py", "run", "Run the application.", "def run():\n    return 1\n")

    text = chunk_to_embedding_text(chunk)

    assert "path: src/app.py" in text
    assert "symbol: run" in text
    assert "docstring: Run the application." in text
    assert "def run" in text


def test_load_or_compute_embeddings_reuses_matching_cache(tmp_path: Path) -> None:
    calls = 0

    def embedder(texts: Sequence[str], batch_size: int) -> list[list[float]]:
        nonlocal calls
        calls += 1
        return [[float(len(text)), 1.0] for text in texts]

    chunk = _chunk("alpha")
    cache_path = tmp_path / "embeddings.pkl"

    first = load_or_compute_embeddings([chunk], cache_path=cache_path, model_name="test-model", embedder=embedder)
    second = load_or_compute_embeddings([chunk], cache_path=cache_path, model_name="test-model", embedder=embedder)

    assert first == second
    assert calls == 1


def test_load_or_compute_embeddings_rebuilds_when_chunks_change(tmp_path: Path) -> None:
    calls = 0

    def embedder(texts: Sequence[str], batch_size: int) -> list[list[float]]:
        nonlocal calls
        calls += 1
        return [[float(calls), 0.0] for _ in texts]

    cache_path = tmp_path / "embeddings.pkl"

    first = load_or_compute_embeddings([_chunk("alpha")], cache_path=cache_path, model_name="test-model", embedder=embedder)
    second = load_or_compute_embeddings([_chunk("beta")], cache_path=cache_path, model_name="test-model", embedder=embedder)

    assert first == [[1.0, 0.0]]
    assert second == [[2.0, 0.0]]
    assert calls == 2


def test_index_chunks_and_query_dense_returns_ranked_metadata(tmp_path: Path) -> None:
    chunks = [
        _chunk("alpha", function_name="alpha_func", source="def alpha_func():\n    return 'alpha'\n"),
        _chunk("beta", function_name="beta_func", source="def beta_func():\n    return 'beta'\n"),
    ]
    collection = FakeCollection()

    dense_index = index_chunks(
        chunks,
        cache_path=tmp_path / "embeddings.pkl",
        collection=collection,
        model_name="test-model",
        embedder=_keyword_embedder,
    )
    results = dense_index.query_dense("beta", k=1)

    assert results == [
        {
            "chunk": {
                "id": "beta",
                "filepath": "src/sample.py",
                "function_name": "beta_func",
                "start_line": 1,
                "end_line": 2,
                "docstring": "",
                "source": "def beta_func():\n    return 'beta'\n",
                "language": "python",
            },
            "score": 1.0,
            "rank": 1,
            "retriever": "dense",
        }
    ]


def test_query_dense_can_attach_neighboring_symbol_chunks(tmp_path: Path) -> None:
    chunks = [
        _chunk("big-1", filepath="src/big.py", function_name="big", source="def big():\n    alpha()\n", start_line=1, end_line=2),
        _chunk("big-2", filepath="src/big.py", function_name="big", source="    beta()\n", start_line=2, end_line=3),
        _chunk("big-3", filepath="src/big.py", function_name="big", source="    return beta\n", start_line=3, end_line=4),
    ]
    collection = FakeCollection()

    dense_index = index_chunks(
        chunks,
        cache_path=tmp_path / "embeddings.pkl",
        collection=collection,
        model_name="test-model",
        embedder=_keyword_embedder,
    )
    results = dense_index.query_dense("beta", k=1, linked_window=1)

    assert results[0]["chunk"]["id"] == "big-2"
    assert [chunk["id"] for chunk in results[0]["linked_chunks"]] == ["big-1", "big-3"]


def test_filter_chunks_by_path_supports_include_and_exclude_prefixes() -> None:
    chunks = [
        _chunk("app", filepath="fastapi/routing.py"),
        _chunk("docs", filepath="docs_src/body/tutorial001.py"),
        _chunk("test", filepath="tests/test_routing.py"),
    ]

    filtered = filter_chunks_by_path(
        chunks,
        include_path_prefixes=["fastapi"],
        exclude_path_prefixes=["fastapi/openapi"],
    )

    assert [chunk.id for chunk in filtered] == ["app"]


def test_filter_chunks_by_path_can_exclude_common_test_paths() -> None:
    chunks = [
        _chunk("source", filepath="fastapi/routing.py"),
        _chunk("tests-dir", filepath="tests/test_routing.py"),
        _chunk("nested-tests-dir", filepath="pkg/tests/test_helpers.py"),
        _chunk("test-file", filepath="fastapi/test_helpers.py"),
    ]

    filtered = filter_chunks_by_path(chunks, exclude_tests=True)

    assert [chunk.id for chunk in filtered] == ["source"]


def test_dense_index_query_respects_non_positive_k() -> None:
    dense_index = DenseIndex(collection=FakeCollection(), embedder=_keyword_embedder, model_name="test-model")

    assert dense_index.query_dense("alpha", k=0) == []


def test_cli_help_is_available_without_dense_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.index_dense", "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "GraphCodeBERT dense Chroma index" in result.stdout


def _chunk(
    chunk_id: str,
    filepath: str = "src/sample.py",
    function_name: str = "sample",
    docstring: str = "",
    source: str = "def sample():\n    return None\n",
    start_line: int = 1,
    end_line: int = 2,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        filepath=filepath,
        function_name=function_name,
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
        source=source,
        language="python",
    )


def _keyword_embedder(texts: Sequence[str], batch_size: int) -> list[list[float]]:
    return [[0.0, 1.0] if "beta" in text.lower() else [1.0, 0.0] for text in texts]


def _dot(left: Sequence[float], right: object) -> float:
    right_vector = right if isinstance(right, Sequence) else []
    return sum(a * b for a, b in zip(left, right_vector))

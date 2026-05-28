import json
import subprocess
import sys
from pathlib import Path

import src.build_index as build_index
from src.bm25_index import fingerprint_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_indexes_uses_one_filtered_chunk_corpus_for_dense_and_bm25(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path / "src" / "app.py", "VALUE = 1\n\n\ndef run():\n    return VALUE\n")
    _write(tmp_path / "tests" / "test_app.py", "def test_run():\n    assert True\n")
    calls: dict[str, object] = {}

    def fake_index_chunks(chunks, **kwargs):
        calls["dense_chunks"] = list(chunks)
        calls["dense_kwargs"] = kwargs
        return "dense-index"

    def fake_build_and_save(chunks, path):
        calls["bm25_chunks"] = list(chunks)
        calls["bm25_path"] = path
        return "bm25-index"

    monkeypatch.setattr(build_index, "index_chunks", fake_index_chunks)
    monkeypatch.setattr(build_index, "build_and_save", fake_build_and_save)

    result = build_index.build_indexes(
        tmp_path,
        cache_path=tmp_path / "embeddings.pkl",
        chroma_path=tmp_path / "chroma",
        bm25_path=tmp_path / "bm25.pkl",
        manifest_path=tmp_path / "manifest.json",
        model_name="test-model",
        exclude_tests=True,
        embedder=lambda texts, batch_size: [[1.0] for _ in texts],
    )

    dense_chunks = calls["dense_chunks"]
    bm25_chunks = calls["bm25_chunks"]
    assert fingerprint_chunks(dense_chunks) == fingerprint_chunks(bm25_chunks)
    assert result["fingerprint"] == fingerprint_chunks(dense_chunks)
    assert all(not chunk.filepath.startswith("tests/") for chunk in dense_chunks)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_fingerprint"] == result["fingerprint"]
    assert manifest["chunk_count"] == len(dense_chunks)
    assert manifest["dense_model"] == "test-model"


def test_build_index_cli_help_is_available_without_embedding_model() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.build_index", "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Build dense Chroma and BM25 indexes" in result.stdout
    assert "--bm25-index-path" in result.stdout
    assert "--manifest-path" in result.stdout


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

import json
import subprocess
import sys
from pathlib import Path

from src.chunker import chunk_repository, make_chunk_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REPO = PROJECT_ROOT / "tests" / "fixtures" / "sample_python_project"


def test_extracts_top_level_function() -> None:
    chunks = _chunks_by_name()

    chunk = chunks["top_level"]

    assert chunk.filepath == "sample.py"
    assert chunk.language == "python"
    assert chunk.docstring == "Add two values."
    assert "def top_level" in chunk.source


def test_extracts_async_function() -> None:
    chunks = _chunks_by_name()

    chunk = chunks["fetch_user"]

    assert chunk.docstring == "Fetch a user identifier asynchronously."
    assert "async def fetch_user" in chunk.source


def test_extracts_class_chunk() -> None:
    chunks = _chunks_by_name()

    chunk = chunks["Greeter"]

    assert chunk.docstring == "Build greeting messages."
    assert "class Greeter" in chunk.source
    assert "def greet" in chunk.source


def test_extracts_methods_with_qualified_names() -> None:
    chunks = _chunks_by_name()

    assert chunks["Greeter.greet"].docstring == "Build a synchronous greeting."
    assert chunks["Greeter.greet_async"].docstring == "Build an asynchronous greeting."


def test_chunk_ids_are_stable() -> None:
    first = chunk_repository(FIXTURE_REPO)
    second = chunk_repository(FIXTURE_REPO)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert make_chunk_id("sample.py", "top_level", 4, 7) == chunks_id(first, "top_level")


def test_cli_prints_summary() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.chunker", "--repo", str(FIXTURE_REPO)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Found 5 chunks" in result.stdout
    assert "Sample chunk" in result.stdout


def test_cli_json_outputs_chunk_dicts() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.chunker", "--repo", str(FIXTURE_REPO), "--json"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert len(payload) == 5
    assert {item["function_name"] for item in payload} >= {"Greeter.greet", "fetch_user"}


def _chunks_by_name():
    return {chunk.function_name: chunk for chunk in chunk_repository(FIXTURE_REPO)}


def chunks_id(chunks, name: str) -> str:
    return next(chunk.id for chunk in chunks if chunk.function_name == name)

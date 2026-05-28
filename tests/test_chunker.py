import json
import subprocess
import sys
from pathlib import Path

from src.chunker import chunk_repository, make_chunk_id, split_oversized_chunk
from src.schema import Chunk

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


def test_extracts_module_level_constants(tmp_path: Path) -> None:
    source = """import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {".py", ".js"}
DEFAULT_ROOT = Path("data")

def run() -> str:
    return os.name
"""
    _write(tmp_path / "settings.py", source)

    chunks = {chunk.function_name: chunk for chunk in chunk_repository(tmp_path)}

    module_chunk = chunks["__module__"]
    assert module_chunk.filepath == "settings.py"
    assert module_chunk.language == "python"
    assert "SUPPORTED_EXTENSIONS" in module_chunk.source
    assert "def run" not in module_chunk.source


def test_extracts_readme_chunks(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "# Project Title\n\nThis project answers code questions.\n\n## Usage\n\nRun the local pipeline.\n",
    )

    chunks = chunk_repository(tmp_path)

    assert [chunk.function_name for chunk in chunks] == ["README:project-title", "README:usage"]
    assert all(chunk.filepath == "README.md" for chunk in chunks)
    assert all(chunk.language == "markdown" for chunk in chunks)
    assert "This project answers code questions." in chunks[0].source
    assert "Run the local pipeline." in chunks[1].source


def test_chunk_ids_are_stable() -> None:
    first = chunk_repository(FIXTURE_REPO)
    second = chunk_repository(FIXTURE_REPO)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert make_chunk_id("sample.py", "top_level", 4, 7) == chunks_id(first, "top_level")


def test_split_oversized_chunk_keeps_symbol_linkage() -> None:
    chunk = Chunk(
        id="original",
        filepath="src/big.py",
        function_name="Big.method",
        start_line=10,
        end_line=17,
        docstring="Large method.",
        source="\n".join(f"line {line}" for line in range(10, 18)),
        language="python",
    )

    parts = split_oversized_chunk(chunk, max_chunk_lines=4, chunk_overlap_lines=1)

    assert [part.function_name for part in parts] == ["Big.method", "Big.method", "Big.method"]
    assert [(part.start_line, part.end_line) for part in parts] == [(10, 13), (13, 16), (16, 17)]
    assert len({part.id for part in parts}) == 3
    assert all(part.docstring == "Large method." for part in parts)


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


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

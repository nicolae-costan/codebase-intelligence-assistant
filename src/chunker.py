"""Semantic chunk extraction for Python repositories.

The chunker uses tree-sitter for symbol boundaries when available and Python's
standard `ast` module for docstring extraction. A small AST boundary fallback is
kept so local tests still produce useful feedback before dependencies are
installed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.schema import Chunk

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
}

PYTHON_LANGUAGE = "python"
DEFINITION_TYPES = {
    "class_definition",
    "function_definition",
    "async_function_definition",
}
CLASS_TYPES = {"class_definition"}
FUNCTION_TYPES = {"function_definition", "async_function_definition"}
DEFAULT_CHUNK_OVERLAP_LINES = 20


@dataclass(frozen=True)
class _SymbolRange:
    """Internal symbol boundary produced by tree-sitter or AST fallback."""

    name: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _LineWindow:
    """Line offsets for one child chunk inside a larger symbol."""

    start: int
    end: int


def chunk_repository(
    repo: str | Path,
    *,
    max_chunk_lines: int | None = None,
    chunk_overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
) -> list[Chunk]:
    """Extract semantic Python chunks from all supported files under `repo`.

    Args:
        repo: Repository or directory path to scan recursively.
        max_chunk_lines: Optional maximum line span before a symbol is split
            into linked overlapping subchunks.
        chunk_overlap_lines: Number of overlapping lines between split chunks.

    Returns:
        Chunks sorted by file path and source location.
    """

    repo_path = Path(repo).resolve()
    chunks: list[Chunk] = []
    for path in iter_python_files(repo_path):
        chunks.extend(
            chunk_file(
                path,
                repo_root=repo_path,
                max_chunk_lines=max_chunk_lines,
                chunk_overlap_lines=chunk_overlap_lines,
            )
        )
    return sorted(chunks, key=lambda chunk: (chunk.filepath, chunk.start_line, chunk.end_line, chunk.function_name))


def iter_python_files(root: str | Path) -> Iterable[Path]:
    """Yield Python files under `root`, skipping generated and cache folders."""

    root_path = Path(root)
    for path in sorted(root_path.rglob("*.py")):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def chunk_file(
    path: str | Path,
    repo_root: str | Path | None = None,
    *,
    max_chunk_lines: int | None = None,
    chunk_overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
) -> list[Chunk]:
    """Extract semantic chunks from one Python file.

    Args:
        path: Python file to parse.
        repo_root: Optional root used to compute repository-relative paths.
        max_chunk_lines: Optional maximum line span before a symbol is split
            into linked overlapping subchunks.
        chunk_overlap_lines: Number of overlapping lines between split chunks.

    Returns:
        Function, async function, class, and method chunks from the file.
    """

    file_path = Path(path).resolve()
    source = file_path.read_text(encoding="utf-8")
    relative_path = _relative_path(file_path, repo_root)
    docstrings = _collect_docstrings(source)
    lines = source.splitlines()

    chunks: list[Chunk] = []
    for symbol in _extract_symbol_ranges(source):
        source_text = "\n".join(lines[symbol.start_line - 1 : symbol.end_line])
        docstring = docstrings.by_qual_and_line.get((symbol.name, symbol.start_line), docstrings.by_qual.get(symbol.name, ""))
        chunk = Chunk(
            id=make_chunk_id(relative_path, symbol.name, symbol.start_line, symbol.end_line),
            filepath=relative_path,
            function_name=symbol.name,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            docstring=docstring,
            source=source_text,
            language=PYTHON_LANGUAGE,
        )
        chunks.extend(
            split_oversized_chunk(
                chunk,
                max_chunk_lines=max_chunk_lines,
                chunk_overlap_lines=chunk_overlap_lines,
            )
        )
    return chunks


def split_oversized_chunk(
    chunk: Chunk,
    *,
    max_chunk_lines: int | None,
    chunk_overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
) -> list[Chunk]:
    """Split a large chunk into overlapping linked parts.

    The returned parts keep the original `filepath` and `function_name`, so a
    retriever can recover neighboring parts from the same symbol after one part
    matches the query.
    """

    if max_chunk_lines is None:
        return [chunk]
    if max_chunk_lines <= 0:
        raise ValueError("max_chunk_lines must be positive when provided.")
    if chunk_overlap_lines < 0:
        raise ValueError("chunk_overlap_lines cannot be negative.")
    if chunk_overlap_lines >= max_chunk_lines:
        raise ValueError("chunk_overlap_lines must be smaller than max_chunk_lines.")

    line_count = chunk.end_line - chunk.start_line + 1
    if line_count <= max_chunk_lines:
        return [chunk]

    source_lines = chunk.source.splitlines()
    windows = _split_line_windows(line_count, max_chunk_lines=max_chunk_lines, chunk_overlap_lines=chunk_overlap_lines)
    parts: list[Chunk] = []
    for part_index, window in enumerate(windows, start=1):
        start_line = chunk.start_line + window.start
        end_line = chunk.start_line + window.end - 1
        source = "\n".join(source_lines[window.start : window.end])
        parts.append(
            Chunk(
                id=make_chunk_id(
                    chunk.filepath,
                    f"{chunk.function_name}:part:{part_index}/{len(windows)}",
                    start_line,
                    end_line,
                ),
                filepath=chunk.filepath,
                function_name=chunk.function_name,
                start_line=start_line,
                end_line=end_line,
                docstring=chunk.docstring,
                source=source,
                language=chunk.language,
            )
        )
    return parts


def make_chunk_id(filepath: str, function_name: str, start_line: int, end_line: int) -> str:
    """Create a deterministic chunk identifier from stable source metadata."""

    key = f"{filepath}:{function_name}:{start_line}:{end_line}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chunker CLI and print either a sample summary or JSON output."""

    parser = argparse.ArgumentParser(description="Extract semantic Python chunks from a repository.")
    parser.add_argument("--repo", required=True, help="Repository or directory path to scan.")
    parser.add_argument("--json", action="store_true", help="Emit all chunks as JSON.")
    parser.add_argument(
        "--max-chunk-lines",
        type=int,
        help="Split symbols longer than this many lines into overlapping linked subchunks.",
    )
    parser.add_argument(
        "--chunk-overlap-lines",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_LINES,
        help="Line overlap between subchunks created by --max-chunk-lines.",
    )
    args = parser.parse_args(argv)

    chunks = chunk_repository(
        args.repo,
        max_chunk_lines=args.max_chunk_lines,
        chunk_overlap_lines=args.chunk_overlap_lines,
    )
    if args.json:
        print(json.dumps([chunk.to_dict() for chunk in chunks], indent=2))
        return 0

    print(f"Found {len(chunks)} chunks in {Path(args.repo)}")
    if chunks:
        sample = chunks[0]
        print()
        print("Sample chunk")
        print(f"  id: {sample.id}")
        print(f"  filepath: {sample.filepath}")
        print(f"  function_name: {sample.function_name}")
        print(f"  lines: {sample.start_line}-{sample.end_line}")
        print(f"  docstring: {sample.docstring or '<none>'}")
        print("  source:")
        for line in sample.source.splitlines()[:12]:
            print(f"    {line}")
    return 0


@dataclass(frozen=True)
class _Docstrings:
    by_qual: dict[str, str]
    by_qual_and_line: dict[tuple[str, int], str]


def _collect_docstrings(source: str) -> _Docstrings:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _Docstrings(by_qual={}, by_qual_and_line={})

    by_qual: dict[str, str] = {}
    by_qual_and_line: dict[tuple[str, int], str] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._record(node)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def _record(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            qualified_name = ".".join([*self.scope, node.name])
            docstring = ast.get_docstring(node, clean=True) or ""
            by_qual.setdefault(qualified_name, docstring)
            by_qual_and_line[(qualified_name, node.lineno)] = docstring

    Visitor().visit(tree)
    return _Docstrings(by_qual=by_qual, by_qual_and_line=by_qual_and_line)


def _extract_symbol_ranges(source: str) -> list[_SymbolRange]:
    parser = _load_tree_sitter_parser()
    if parser is None:
        return _extract_symbol_ranges_with_ast(source)

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    symbols: list[_SymbolRange] = []

    def text_for(node: object) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8")

    def visit(node: object, scope: list[str]) -> None:
        if node.type == "decorated_definition":
            definition = next((child for child in node.children if child.type in DEFINITION_TYPES), None)
            if definition is not None:
                visit_definition(definition, node, scope)
            return

        if node.type in DEFINITION_TYPES:
            visit_definition(node, node, scope)
            return

        for child in node.children:
            visit(child, scope)

    def visit_definition(definition: object, range_node: object, scope: list[str]) -> None:
        name_node = definition.child_by_field_name("name")
        if name_node is None:
            return

        name = text_for(name_node)
        qualified_name = ".".join([*scope, name])
        symbols.append(
            _SymbolRange(
                name=qualified_name,
                start_line=range_node.start_point[0] + 1,
                end_line=range_node.end_point[0] + 1,
            )
        )

        next_scope = [*scope, name]
        for child in definition.children:
            visit(child, next_scope)

    visit(tree.root_node, [])
    return symbols


def _extract_symbol_ranges_with_ast(source: str) -> list[_SymbolRange]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    symbols: list[_SymbolRange] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._record(node)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def _record(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            if node.end_lineno is None:
                return
            symbols.append(
                _SymbolRange(
                    name=".".join([*self.scope, node.name]),
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                )
            )

    Visitor().visit(tree)
    return symbols


def _split_line_windows(line_count: int, *, max_chunk_lines: int, chunk_overlap_lines: int) -> list[_LineWindow]:
    windows: list[_LineWindow] = []
    start = 0
    while start < line_count:
        end = min(start + max_chunk_lines, line_count)
        windows.append(_LineWindow(start=start, end=end))
        if end == line_count:
            break
        start = end - chunk_overlap_lines
    return windows


def _load_tree_sitter_parser() -> object | None:
    try:
        from tree_sitter_languages import get_parser
    except ImportError:
        return None

    try:
        return get_parser(PYTHON_LANGUAGE)
    except Exception:
        return None


def _relative_path(file_path: Path, repo_root: str | Path | None) -> str:
    if repo_root is None:
        return file_path.name

    root = Path(repo_root).resolve()
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())

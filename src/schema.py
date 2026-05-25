"""Shared data structures for the codebase intelligence pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    """A semantic source-code unit ready for retrieval and citation.

    Attributes:
        id: Stable hash derived from path, symbol name, and line range.
        filepath: Repository-relative path using POSIX separators.
        function_name: Qualified symbol name, such as `Class.method`.
        start_line: 1-based inclusive start line.
        end_line: 1-based inclusive end line.
        docstring: Cleaned docstring for the symbol, or an empty string.
        source: Raw source text covered by the chunk.
        language: Source language identifier, currently `python`.
    """

    id: str
    filepath: str
    function_name: str
    start_line: int
    end_line: int
    docstring: str
    source: str
    language: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the chunk."""

        return asdict(self)

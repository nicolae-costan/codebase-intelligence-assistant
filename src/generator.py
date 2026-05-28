"""Grounded answer generation through Ollama's OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.schema import Chunk

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MAX_CONTEXT_CHARS = 24000
DEFAULT_MAX_CHUNK_CHARS = 8000
REFUSAL_ANSWER = "I cannot find this in the codebase."

SYSTEM_PROMPT = """You are a codebase assistant. Answer questions strictly using the provided source code snippets.
If the answer is not present in the provided context, say "I cannot find this in the codebase."
Do not invent function names, libraries, or logic not shown in the context.
Do not give general background about the project or framework.
Only mention files, functions, classes, and behavior that appear in the snippets.
Every factual sentence in the answer must include at least one snippet citation like [1] or [2].

When answering:
1. Directly answer the question in 2-6 concise bullets.
2. Cite the specific snippet label you are drawing from in each bullet.
3. If the snippets are only partially relevant, say exactly what can and cannot be determined."""
_CITATION_RE = re.compile(r"\[\d+\]")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


@dataclass(frozen=True)
class _NormalizedChunk:
    filepath: str
    function_name: str
    start_line: int
    end_line: int
    docstring: str
    source: str


def format_context_chunks(
    context_chunks: Sequence[Chunk | Mapping[str, object]],
    *,
    max_context_chars: int | None = DEFAULT_MAX_CONTEXT_CHARS,
    max_chunk_chars: int | None = DEFAULT_MAX_CHUNK_CHARS,
) -> str:
    """Render chunks into a cited source context for the generator."""

    context, _count = _render_context_chunks(
        context_chunks,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
    )
    return context


def _render_context_chunks(
    context_chunks: Sequence[Chunk | Mapping[str, object]],
    *,
    max_context_chars: int | None,
    max_chunk_chars: int | None,
) -> tuple[str, int]:
    rendered: list[str] = []
    for index, raw_chunk in enumerate(context_chunks, start=1):
        chunk = _normalize_chunk(raw_chunk)
        header = f"[{index}] {chunk.filepath}::{chunk.function_name} lines {chunk.start_line}-{chunk.end_line}"
        parts = [header]
        if chunk.docstring:
            parts.append(f"Docstring: {chunk.docstring}")
        parts.append("Source:")
        parts.append(_limit_text(chunk.source, max_chunk_chars))
        block = "\n".join(parts)
        candidate = "\n\n---\n\n".join([*rendered, block]) if rendered else block
        if max_context_chars is not None and len(candidate) > max_context_chars:
            if not rendered:
                rendered.append(_limit_text(block, max_context_chars))
            break
        rendered.append(block)
    return "\n\n---\n\n".join(rendered), len(rendered)


def generate(
    context_chunks: Sequence[Chunk | Mapping[str, object]],
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key: str | None = None,
    client: Any | None = None,
    max_context_chars: int | None = DEFAULT_MAX_CONTEXT_CHARS,
    max_chunk_chars: int | None = DEFAULT_MAX_CHUNK_CHARS,
) -> str:
    """Generate a grounded answer from supplied source chunks."""

    if not context_chunks:
        return REFUSAL_ANSWER

    context, context_count = _render_context_chunks(
        context_chunks,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
    )
    if context_count == 0:
        return REFUSAL_ANSWER

    resolved_client = client or _make_openai_client(base_url=base_url, api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(context, query)},
    ]
    response = resolved_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        return REFUSAL_ANSWER
    stripped_answer = answer.strip()
    if not _CITATION_RE.search(stripped_answer):
        return REFUSAL_ANSWER
    if _has_invalid_citations(stripped_answer, context_count):
        return REFUSAL_ANSWER
    if _missing_required_identifiers(query, stripped_answer):
        return REFUSAL_ANSWER
    return stripped_answer


def main(argv: Sequence[str] | None = None) -> int:
    """Run a hardcoded-context generator smoke test."""

    parser = argparse.ArgumentParser(description="Generate a grounded answer with Ollama/Qwen.")
    parser.add_argument("--query", required=True, help="Question to answer from the provided context.")
    parser.add_argument("--fixture", action="store_true", help="Use a small built-in fixture context.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OLLAMA_BASE_URL or localhost.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON object with the answer.")
    args = parser.parse_args(argv)

    if not args.fixture:
        parser.error("Only --fixture context is currently supported by this standalone T8 CLI.")

    answer = generate(
        _fixture_chunks(),
        args.query,
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
    )
    if args.json:
        print(json.dumps({"answer": answer}, indent=2))
    else:
        print(answer)
    return 0


def _build_user_message(context: str, query: str) -> str:
    return f"""Question:
{query}

Source snippets:
{context}

Answer strictly from the source snippets above. Use snippet citations like [1] in every bullet."""


def _limit_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= len("\n...[truncated]"):
        return text[:max_chars]
    return text[: max_chars - len("\n...[truncated]")].rstrip() + "\n...[truncated]"


def _has_invalid_citations(answer: str, context_count: int) -> bool:
    citations = [int(match[1:-1]) for match in _CITATION_RE.findall(answer)]
    return any(citation < 1 or citation > context_count for citation in citations)


def _missing_required_identifiers(query: str, answer: str) -> bool:
    required_identifiers = _required_query_identifiers(query)
    if not required_identifiers:
        return False
    answer_lower = answer.lower()
    return any(identifier.lower() not in answer_lower for identifier in required_identifiers)


def _required_query_identifiers(query: str) -> list[str]:
    identifiers: list[str] = []
    for token in _IDENTIFIER_RE.findall(query):
        if _looks_like_code_identifier(token):
            identifiers.append(token)
    return identifiers


def _looks_like_code_identifier(token: str) -> bool:
    if "_" in token:
        return True
    return bool(re.search(r"[a-z][A-Z]", token))


def _make_openai_client(*, base_url: str | None = None, api_key: str | None = None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Ollama generation requires the `openai` package. "
            "Install project dependencies with `python -m pip install -e .`."
        ) from exc

    return OpenAI(
        base_url=base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key or os.environ.get("OLLAMA_API_KEY", DEFAULT_API_KEY),
    )


def _normalize_chunk(raw_chunk: Chunk | Mapping[str, object]) -> _NormalizedChunk:
    if isinstance(raw_chunk, Chunk):
        return _NormalizedChunk(
            filepath=raw_chunk.filepath,
            function_name=raw_chunk.function_name,
            start_line=raw_chunk.start_line,
            end_line=raw_chunk.end_line,
            docstring=raw_chunk.docstring,
            source=raw_chunk.source,
        )

    if isinstance(raw_chunk, Mapping):
        return _NormalizedChunk(
            filepath=str(raw_chunk.get("filepath", "")),
            function_name=str(raw_chunk.get("function_name", "")),
            start_line=int(raw_chunk.get("start_line", 0)),
            end_line=int(raw_chunk.get("end_line", 0)),
            docstring=str(raw_chunk.get("docstring", "")),
            source=str(raw_chunk.get("source", "")),
        )

    raise TypeError(f"Unsupported context chunk type: {type(raw_chunk).__name__}")


def _fixture_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="fixture-greeter-greet",
            filepath="tests/fixtures/sample_python_project/sample.py",
            function_name="Greeter.greet",
            start_line=21,
            end_line=24,
            docstring="Build a synchronous greeting.",
            source='def greet(self, name: str) -> str:\n    """Build a synchronous greeting."""\n\n    return f"{self.prefix}, {name}!"',
            language="python",
        )
    ]


if __name__ == "__main__":
    raise SystemExit(main())

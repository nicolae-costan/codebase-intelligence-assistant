"""T7 RAG orchestration over hybrid retrieval and grounded generation."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.confidence import DEFAULT_CONFIDENCE_LOG_PATH, DEFAULT_CONFIDENCE_THRESHOLD
from src.generator import REFUSAL_ANSWER
from src.schema import Chunk

DEFAULT_TOP_K = 7
DEFAULT_LOG_PATH = Path("results/iterative_retrieval.jsonl")
VALID_MODES = {"single", "iterative"}

RetrieverFn = Callable[[str, int], Sequence[Any]]
GeneratorFn = Callable[..., str]
ConfidenceEstimatorFn = Callable[..., Any]


def iterative_rag(
    query: str,
    mode: str = "single",
    top_k: int = DEFAULT_TOP_K,
    *,
    retriever: RetrieverFn | None = None,
    generator_fn: GeneratorFn | None = None,
    log_path: str | Path | None = DEFAULT_LOG_PATH,
    confidence: bool = False,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    confidence_log_path: str | Path | None = DEFAULT_CONFIDENCE_LOG_PATH,
    confidence_estimator: ConfidenceEstimatorFn | None = None,
) -> dict[str, object]:
    """Run single-pass or two-pass iterative RAG for a user query.

    The public output is shaped for ``eval.run_eval``: the final answer is in
    ``answer`` and the final retrieval context is in ``retrieved_chunks``.
    """

    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported pipeline mode {mode!r}. Expected one of: {', '.join(sorted(VALID_MODES))}.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    retriever_fn = retriever or _load_default_retriever()
    generate_fn = generator_fn or _load_default_generator()
    total_start = time.perf_counter()

    pass_1 = _run_pass(
        retriever_fn=retriever_fn,
        generator_fn=generate_fn,
        retrieval_query=query,
        generation_query=query,
        top_k=top_k,
    )

    trace: dict[str, object] = {"pass_1": pass_1.trace}
    result: dict[str, object] = {
        "answer": pass_1.answer,
        "retrieved_chunks": pass_1.chunks,
        "trace": trace,
        "mode": mode,
    }

    if mode == "iterative" and pass_1.answer != REFUSAL_ANSWER:
        pass_2 = iterative_pass(
            query=query,
            partial_answer=pass_1.answer,
            top_k=top_k,
            retriever=retriever_fn,
            generator_fn=generate_fn,
        )
        trace["pass_2"] = pass_2.trace
        result["answer"] = pass_2.answer
        result["retrieved_chunks"] = pass_2.chunks
        result["partial_answer"] = pass_1.answer
    elif mode == "iterative":
        trace["pass_2"] = {"skipped": "pass_1_refused"}
        result["partial_answer"] = pass_1.answer

    if confidence:
        _apply_confidence(
            result,
            query=query,
            generator_fn=generate_fn,
            threshold=confidence_threshold,
            log_path=confidence_log_path,
            confidence_estimator=confidence_estimator,
        )

    result["trace"] = trace | {"total_ms": _elapsed_ms(total_start)}
    
    # Apply grounding check
    from src.grounding import check_grounding
    grounding_result = check_grounding(str(result.get("answer", "")), result.get("retrieved_chunks", []))
    result["grounded"] = grounding_result["grounded"]
    result["ungrounded_claims"] = grounding_result["ungrounded_claims"]

    _append_log_record(result, query=query, top_k=top_k, log_path=log_path)
    return result


def single_pass_rag(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    retriever: RetrieverFn | None = None,
    generator_fn: GeneratorFn | None = None,
    log_path: str | Path | None = DEFAULT_LOG_PATH,
    confidence: bool = False,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    confidence_log_path: str | Path | None = DEFAULT_CONFIDENCE_LOG_PATH,
    confidence_estimator: ConfidenceEstimatorFn | None = None,
) -> dict[str, object]:
    """Run the pass-1-only RAG ablation."""

    return iterative_rag(
        query,
        mode="single",
        top_k=top_k,
        retriever=retriever,
        generator_fn=generator_fn,
        log_path=log_path,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
        confidence_log_path=confidence_log_path,
        confidence_estimator=confidence_estimator,
    )


def iterative_pass(
    *,
    query: str,
    partial_answer: str,
    top_k: int,
    retriever: RetrieverFn,
    generator_fn: GeneratorFn,
) -> _PassResult:
    """Run the second RepoCoder-style retrieval and final generation pass."""

    refined_query = f"{query}\n\nPartial answer:\n{partial_answer}"
    return _run_pass(
        retriever_fn=retriever,
        generator_fn=generator_fn,
        retrieval_query=refined_query,
        generation_query=query,
        top_k=top_k,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for T7 RAG orchestration."""

    parser = argparse.ArgumentParser(description="Run single-pass or iterative RAG over the indexed codebase.")
    parser.add_argument("--query", required=True, help="Question to answer from retrieved source context.")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="single", help="RAG orchestration mode.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve per pass.")
    parser.add_argument("--confidence", action="store_true", help="Estimate answer confidence with HonestCoder.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score before flagging a low-confidence answer.",
    )
    parser.add_argument(
        "--confidence-log-path",
        default=str(DEFAULT_CONFIDENCE_LOG_PATH),
        help="JSONL confidence log path. Use 'none' to disable confidence logging.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full pipeline result as JSON.")
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help="JSONL experiment log path. Use 'none' to disable logging.",
    )
    args = parser.parse_args(argv)

    log_path: str | None = args.log_path
    if log_path.lower() in {"", "none", "null"}:
        log_path = None

    confidence_log_path: str | None = args.confidence_log_path
    if confidence_log_path.lower() in {"", "none", "null"}:
        confidence_log_path = None

    try:
        result = iterative_rag(
            args.query,
            mode=args.mode,
            top_k=args.top_k,
            log_path=log_path,
            confidence=args.confidence,
            confidence_threshold=args.confidence_threshold,
            confidence_log_path=confidence_log_path,
        )
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["answer"])
    return 0


@dataclass(frozen=True)
class _PassResult:
    answer: str
    chunks: list[dict[str, object]]
    trace: dict[str, object]


def _run_pass(
    *,
    retriever_fn: RetrieverFn,
    generator_fn: GeneratorFn,
    retrieval_query: str,
    generation_query: str,
    top_k: int,
) -> _PassResult:
    retrieval_start = time.perf_counter()
    raw_results = retriever_fn(retrieval_query, top_k)
    retrieval_ms = _elapsed_ms(retrieval_start)
    chunks = _normalize_retrieved_chunks(raw_results)

    generation_start = time.perf_counter()
    answer = generator_fn(chunks, generation_query)
    generation_ms = _elapsed_ms(generation_start)

    return _PassResult(
        answer=answer,
        chunks=chunks,
        trace={
            "query": retrieval_query,
            "retrieved_chunks": chunks,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
        },
    )


def _apply_confidence(
    result: dict[str, object],
    *,
    query: str,
    generator_fn: GeneratorFn,
    threshold: float,
    log_path: str | Path | None,
    confidence_estimator: ConfidenceEstimatorFn | None,
) -> None:
    estimator = confidence_estimator or _load_default_confidence_estimator()
    confidence_result = estimator(
        query=query,
        context_chunks=result.get("retrieved_chunks", []),
        generator_fn=generator_fn,
        threshold=threshold,
        log_path=log_path,
    )
    details = _confidence_result_to_dict(confidence_result)
    candidate_answer = details.get("answer", result.get("answer", ""))
    should_refuse = bool(details.get("should_refuse", False))
    if should_refuse:
        result["answer"] = REFUSAL_ANSWER
        result["confidence_candidate_answer"] = candidate_answer
    else:
        result["answer"] = candidate_answer
    result["confidence_score"] = details.get("confidence", 0.0)
    result["confidence_level"] = details.get("level", "low")
    result["low_confidence"] = should_refuse
    if details.get("warning"):
        result["confidence_warning"] = details["warning"]
    result["confidence_details"] = details


def _confidence_result_to_dict(confidence_result: object) -> dict[str, object]:
    if isinstance(confidence_result, Mapping):
        return dict(confidence_result)
    to_dict = getattr(confidence_result, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise TypeError(f"Unsupported confidence result type: {type(confidence_result).__name__}")


def _normalize_retrieved_chunks(raw_results: Sequence[Any]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_result in raw_results:
        for payload_index, raw_chunk in enumerate(_iter_retrieved_chunk_payloads(raw_result)):
            chunk = _normalize_chunk_result(raw_chunk)
            if payload_index == 0:
                _attach_retrieval_debug(chunk, raw_result)
            key = _chunk_dedupe_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            chunks.append(chunk)
    return chunks


def _iter_retrieved_chunk_payloads(raw_result: Any) -> Sequence[Any]:
    if isinstance(raw_result, Mapping) and "chunk" in raw_result:
        payloads = [raw_result["chunk"]]
        linked_chunks = raw_result.get("linked_chunks", [])
        if isinstance(linked_chunks, Sequence) and not isinstance(linked_chunks, (str, bytes)):
            payloads.extend(linked_chunks)
        return payloads
    return [raw_result]


def _normalize_chunk_result(raw_result: Any) -> dict[str, object]:
    if isinstance(raw_result, Chunk):
        return raw_result.to_dict()

    if isinstance(raw_result, Mapping):
        if "chunk" in raw_result:
            return _normalize_chunk_result(raw_result["chunk"])
        return _chunk_mapping_to_dict(raw_result)

    raise TypeError(f"Unsupported retriever result type: {type(raw_result).__name__}")


def _attach_retrieval_debug(chunk: dict[str, object], raw_result: Any) -> None:
    if not isinstance(raw_result, Mapping):
        return
    debug = raw_result.get("retrieval_debug")
    if isinstance(debug, Mapping):
        chunk["retrieval_debug"] = dict(debug)


def _chunk_mapping_to_dict(raw_chunk: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": str(_first_present(raw_chunk, "id", "chunk_id", default="")),
        "filepath": str(_first_present(raw_chunk, "filepath", "path", "file", default="")),
        "function_name": str(_first_present(raw_chunk, "function_name", "symbol_name", "name", default="")),
        "start_line": _safe_int(_first_present(raw_chunk, "start_line", default=0)),
        "end_line": _safe_int(_first_present(raw_chunk, "end_line", default=0)),
        "docstring": str(_first_present(raw_chunk, "docstring", default="")),
        "source": str(_first_present(raw_chunk, "source", "text", "document", default="")),
        "language": str(_first_present(raw_chunk, "language", default="")),
    }


def _chunk_dedupe_key(chunk: Mapping[str, object]) -> str:
    chunk_id = str(chunk.get("id", ""))
    if chunk_id:
        return f"id:{chunk_id}"
    return (
        "loc:"
        f"{chunk.get('filepath', '')}:"
        f"{chunk.get('function_name', '')}:"
        f"{chunk.get('start_line', 0)}:"
        f"{chunk.get('end_line', 0)}"
    )


def _first_present(mapping: Mapping[str, object], *keys: str, default: object) -> object:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_default_retriever() -> RetrieverFn:
    try:
        from src.retriever import hybrid_search
    except ModuleNotFoundError as exc:
        if exc.name == "src.retriever":
            raise RuntimeError(
                "T6 hybrid retrieval is unavailable. Implement src.retriever.hybrid_search(query, top_k) first."
            ) from exc
        raise
    except ImportError as exc:
        raise RuntimeError(
            "T6 hybrid retrieval is unavailable. Implement src.retriever.hybrid_search(query, top_k) first."
        ) from exc

    if not callable(hybrid_search):
        raise RuntimeError("T6 hybrid retrieval is unavailable: src.retriever.hybrid_search is not callable.")
    return hybrid_search


def _load_default_generator() -> GeneratorFn:
    from src.generator import generate

    return generate


def _load_default_confidence_estimator() -> ConfidenceEstimatorFn:
    from src.confidence import estimate_confidence

    return estimate_confidence


def _append_log_record(result: dict[str, object], *, query: str, top_k: int, log_path: str | Path | None) -> None:
    if log_path is None:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_log_record(result, query=query, top_k=top_k)) + "\n")


def _log_record(result: dict[str, object], *, query: str, top_k: int) -> dict[str, object]:
    trace = result.get("trace", {})
    pass_1 = trace.get("pass_1", {}) if isinstance(trace, Mapping) else {}
    pass_2 = trace.get("pass_2", {}) if isinstance(trace, Mapping) else {}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": result.get("mode"),
        "query": query,
        "top_k": top_k,
        "pass_1": _log_pass(pass_1),
        "pass_2": _log_pass(pass_2) if pass_2 else None,
        "partial_answer": result.get("partial_answer"),
        "answer": result.get("answer"),
        "latency_ms": trace.get("total_ms") if isinstance(trace, Mapping) else None,
    }


def _log_pass(pass_trace: object) -> dict[str, object]:
    if not isinstance(pass_trace, Mapping):
        return {}
    chunks = pass_trace.get("retrieved_chunks", [])
    return {
        "query": pass_trace.get("query"),
        "retrieved_chunks": [_chunk_summary(chunk) for chunk in chunks] if isinstance(chunks, Sequence) else [],
        "retrieval_ms": pass_trace.get("retrieval_ms"),
        "generation_ms": pass_trace.get("generation_ms"),
    }


def _chunk_summary(chunk: object) -> dict[str, object]:
    if isinstance(chunk, Mapping):
        return {
            "id": chunk.get("id", ""),
            "filepath": chunk.get("filepath", ""),
            "function_name": chunk.get("function_name", ""),
            "start_line": chunk.get("start_line", 0),
            "end_line": chunk.get("end_line", 0),
        }
    return {}


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


if __name__ == "__main__":
    raise SystemExit(main())

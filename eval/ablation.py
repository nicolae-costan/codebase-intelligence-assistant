"""T12 ablation experiments for retrieval and iterative RAG variants."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.run_eval import DEFAULT_TEST_SET_PATH, REFUSAL_MARKERS, load_test_set, run_eval
from src.confidence import DEFAULT_CONFIDENCE_THRESHOLD
from src.pipeline import DEFAULT_TOP_K, iterative_rag

DEFAULT_JSON_PATH = Path("results/ablation.json")
DEFAULT_CSV_PATH = Path("results/ablation.csv")
DEFAULT_PIPELINE_LOG_PATH = Path("results/ablation_runs.jsonl")
DEFAULT_CONFIDENCE_LOG_PATH = Path("results/ablation_confidence_log.jsonl")
DEFAULT_LINKED_WINDOW = 1

DEFAULT_ADVERSARIAL_QUERIES = (
    "Which npm script deploys the React frontend to production?",
    "Where is the Stripe webhook signature verified?",
    "How does the Kubernetes operator rotate TLS certificates?",
)

RetrieverFn = Callable[[str, int], Sequence[Any]]
RetrieverFactory = Callable[[str], RetrieverFn]
PipelineRunner = Callable[..., Any]


@dataclass(frozen=True)
class AblationCondition:
    """One retrieval/pass-count variant in the T12 ablation table."""

    id: str
    label: str
    retrieval: str
    retrieval_passes: str
    mode: str


DEFAULT_CONDITIONS = (
    AblationCondition(
        id="A",
        label="Dense only, single-pass",
        retrieval="dense",
        retrieval_passes="single-pass",
        mode="single",
    ),
    AblationCondition(
        id="B",
        label="Hybrid, single-pass",
        retrieval="hybrid",
        retrieval_passes="single-pass",
        mode="single",
    ),
    AblationCondition(
        id="C",
        label="Dense only, iterative",
        retrieval="dense",
        retrieval_passes="iterative (2-pass)",
        mode="iterative",
    ),
    AblationCondition(
        id="D",
        label="Hybrid, iterative",
        retrieval="hybrid",
        retrieval_passes="iterative (2-pass)",
        mode="iterative",
    ),
)

CSV_BASE_FIELDS = (
    "condition_id",
    "label",
    "retrieval",
    "retrieval_passes",
    "mode",
    "total_questions",
    "answerable_questions",
    "refusal_questions",
    "context_hit_rate",
    "context_recall",
    "context_precision",
    "answer_relevance_proxy",
    "refusal_accuracy",
    "adversarial_queries",
    "adversarial_refusal_rate",
    "avg_confidence",
    "low_confidence_rate",
)


def run_ablation(
    *,
    test_set_path: str | Path = DEFAULT_TEST_SET_PATH,
    output_json_path: str | Path = DEFAULT_JSON_PATH,
    output_csv_path: str | Path = DEFAULT_CSV_PATH,
    top_k: int = DEFAULT_TOP_K,
    use_ragas: bool = False,
    confidence: bool = True,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    pipeline_log_path: str | Path | None = DEFAULT_PIPELINE_LOG_PATH,
    confidence_log_path: str | Path | None = DEFAULT_CONFIDENCE_LOG_PATH,
    adversarial_queries: Sequence[str] | None = None,
    linked_window: int = DEFAULT_LINKED_WINDOW,
    conditions: Sequence[AblationCondition] = DEFAULT_CONDITIONS,
    retriever_factory: RetrieverFactory | None = None,
    pipeline_runner: PipelineRunner = iterative_rag,
) -> dict[str, Any]:
    """Run all T12 ablation conditions and persist JSON plus CSV artifacts."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    test_set = Path(test_set_path)
    resolved_adversarial_queries = _resolve_adversarial_queries(test_set, adversarial_queries)
    make_retriever = retriever_factory or (
        lambda retrieval: make_retriever_for_condition(retrieval, linked_window=linked_window)
    )

    condition_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for condition in conditions:
        eval_outputs: list[Any] = []
        pipeline_fn = _make_condition_pipeline(
            condition,
            top_k=top_k,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
            pipeline_log_path=pipeline_log_path,
            confidence_log_path=confidence_log_path,
            retriever=make_retriever(condition.retrieval),
            pipeline_runner=pipeline_runner,
            captured_outputs=eval_outputs,
        )
        metrics = run_eval(pipeline_fn, test_set, use_ragas=use_ragas)

        adversarial_outputs: list[Any] = []
        adversarial = _run_adversarial_queries(
            condition,
            queries=resolved_adversarial_queries,
            top_k=top_k,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
            pipeline_log_path=pipeline_log_path,
            confidence_log_path=confidence_log_path,
            retriever=make_retriever(condition.retrieval),
            pipeline_runner=pipeline_runner,
            captured_outputs=adversarial_outputs,
        )
        summary = _summarize_condition(metrics, eval_outputs=eval_outputs, adversarial=adversarial)

        result = {
            "condition": asdict(condition),
            "metrics": metrics,
            "adversarial": adversarial,
            "summary": summary,
        }
        condition_results.append(result)
        csv_rows.append(_csv_row(condition, summary, metrics))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_set_path": str(test_set),
        "top_k": top_k,
        "confidence_enabled": confidence,
        "confidence_threshold": confidence_threshold,
        "ragas_enabled": use_ragas,
        "adversarial_queries": list(resolved_adversarial_queries),
        "conditions": condition_results,
    }
    _write_json(payload, output_json_path)
    _write_csv(csv_rows, output_csv_path)
    return payload


def make_retriever_for_condition(retrieval: str, *, linked_window: int = DEFAULT_LINKED_WINDOW) -> RetrieverFn:
    """Build the real dense-only or hybrid retriever used by a condition."""

    if retrieval == "dense":

        def dense_retriever(query: str, top_k: int) -> Sequence[Any]:
            from src.index_dense import query_dense

            return query_dense(query, k=top_k, linked_window=linked_window)

        return dense_retriever

    if retrieval == "hybrid":

        def hybrid_retriever(query: str, top_k: int) -> Sequence[Any]:
            from src.retriever import hybrid_search

            return hybrid_search(query, top_k=top_k, linked_window=linked_window)

        return hybrid_retriever

    raise ValueError(f"Unsupported retrieval condition: {retrieval!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for T12 ablation runs."""

    parser = argparse.ArgumentParser(description="Run T12 ablation experiments and write JSON/CSV result tables.")
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET_PATH), help="Path to eval/test_set.json.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve per pass.")
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_PATH), help="Destination JSON artifact.")
    parser.add_argument("--csv-output", default=str(DEFAULT_CSV_PATH), help="Destination CSV artifact.")
    parser.add_argument("--ragas", action="store_true", help="Also attempt optional RAGAS scoring for each condition.")
    parser.add_argument("--no-confidence", action="store_true", help="Disable HonestCoder confidence/refusal scoring.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score before HonestCoder flags a low-confidence answer.",
    )
    parser.add_argument(
        "--pipeline-log-path",
        default=str(DEFAULT_PIPELINE_LOG_PATH),
        help="JSONL pipeline trace path. Use 'none' to disable.",
    )
    parser.add_argument(
        "--confidence-log-path",
        default=str(DEFAULT_CONFIDENCE_LOG_PATH),
        help="JSONL confidence log path. Use 'none' to disable.",
    )
    parser.add_argument(
        "--adversarial-query",
        action="append",
        default=None,
        help="Out-of-scope query to include. Repeat to replace the default adversarial set.",
    )
    parser.add_argument(
        "--linked-window",
        type=int,
        default=DEFAULT_LINKED_WINDOW,
        help="Neighboring dense subchunks to include around matched chunks.",
    )
    args = parser.parse_args(argv)

    payload = run_ablation(
        test_set_path=args.test_set,
        output_json_path=args.json_output,
        output_csv_path=args.csv_output,
        top_k=args.top_k,
        use_ragas=args.ragas,
        confidence=not args.no_confidence,
        confidence_threshold=args.confidence_threshold,
        pipeline_log_path=_optional_path(args.pipeline_log_path),
        confidence_log_path=_optional_path(args.confidence_log_path),
        adversarial_queries=args.adversarial_query,
        linked_window=args.linked_window,
    )
    _print_summary(payload, json_path=Path(args.json_output), csv_path=Path(args.csv_output))
    return 0


def _make_condition_pipeline(
    condition: AblationCondition,
    *,
    top_k: int,
    confidence: bool,
    confidence_threshold: float,
    pipeline_log_path: str | Path | None,
    confidence_log_path: str | Path | None,
    retriever: RetrieverFn,
    pipeline_runner: PipelineRunner,
    captured_outputs: list[Any],
) -> Callable[[str], Any]:
    def pipeline_fn(question: str) -> Any:
        output = pipeline_runner(
            question,
            mode=condition.mode,
            top_k=top_k,
            retriever=retriever,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
            log_path=pipeline_log_path,
            confidence_log_path=confidence_log_path,
        )
        captured_outputs.append(output)
        return output

    return pipeline_fn


def _run_adversarial_queries(
    condition: AblationCondition,
    *,
    queries: Sequence[str],
    top_k: int,
    confidence: bool,
    confidence_threshold: float,
    pipeline_log_path: str | Path | None,
    confidence_log_path: str | Path | None,
    retriever: RetrieverFn,
    pipeline_runner: PipelineRunner,
    captured_outputs: list[Any],
) -> dict[str, Any]:
    if not queries:
        return {"query_count": 0, "refusal_rate": None, "records": []}

    pipeline_fn = _make_condition_pipeline(
        condition,
        top_k=top_k,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
        pipeline_log_path=pipeline_log_path,
        confidence_log_path=confidence_log_path,
        retriever=retriever,
        pipeline_runner=pipeline_runner,
        captured_outputs=captured_outputs,
    )
    records = [_adversarial_record(query, pipeline_fn(query)) for query in queries]
    return {
        "query_count": len(records),
        "refusal_rate": _average(record["correctly_refused_or_flagged"] for record in records),
        "records": records,
    }


def _adversarial_record(query: str, output: Any) -> dict[str, Any]:
    answer = _answer_from_output(output)
    record = {
        "query": query,
        "answer": answer,
        "correctly_refused_or_flagged": _is_refusal_or_flagged(output),
    }
    if isinstance(output, Mapping):
        record.update(
            {
                "confidence_score": output.get("confidence_score"),
                "confidence_level": output.get("confidence_level"),
                "low_confidence": output.get("low_confidence"),
                "grounded": output.get("grounded"),
                "ungrounded_claims": list(output.get("ungrounded_claims", [])),
                "retrieved_contexts": [_chunk_summary(chunk) for chunk in output.get("retrieved_chunks", [])],
            }
        )
    return record


def _resolve_adversarial_queries(test_set_path: Path, explicit_queries: Sequence[str] | None) -> list[str]:
    if explicit_queries is not None:
        return [query for query in explicit_queries if query.strip()]

    expected_refusal_queries = [
        item.question for item in load_test_set(test_set_path) if item.expected_refusal and item.question.strip()
    ]
    queries = [*expected_refusal_queries, *DEFAULT_ADVERSARIAL_QUERIES]
    return list(dict.fromkeys(queries))


def _summarize_condition(
    metrics: Mapping[str, Any],
    *,
    eval_outputs: Sequence[Any],
    adversarial: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "total_questions": metrics.get("total_questions"),
        "answerable_questions": metrics.get("answerable_questions"),
        "refusal_questions": metrics.get("refusal_questions"),
        "context_hit_rate": metrics.get("context_hit_rate"),
        "context_recall": metrics.get("context_recall"),
        "context_precision": metrics.get("context_precision"),
        "answer_relevance_proxy": metrics.get("answer_relevance_proxy"),
        "refusal_accuracy": metrics.get("refusal_accuracy"),
        "adversarial_queries": adversarial.get("query_count", 0),
        "adversarial_refusal_rate": adversarial.get("refusal_rate"),
        "avg_confidence": _average(_confidence_score(output) for output in eval_outputs),
        "low_confidence_rate": _average(_low_confidence(output) for output in eval_outputs),
    }
    summary.update(_ragas_summary(metrics.get("ragas")))
    return summary


def _csv_row(
    condition: AblationCondition,
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "condition_id": condition.id,
        "label": condition.label,
        "retrieval": condition.retrieval,
        "retrieval_passes": condition.retrieval_passes,
        "mode": condition.mode,
        **summary,
    }
    row.update(_ragas_summary(metrics.get("ragas")))
    return row


def _ragas_summary(raw_ragas: Any) -> dict[str, Any]:
    if not isinstance(raw_ragas, Mapping):
        return {}
    if raw_ragas.get("enabled") is False:
        return {
            "ragas_enabled": False,
            "ragas_reason": raw_ragas.get("reason", ""),
        }
    scores = raw_ragas.get("scores")
    if not isinstance(scores, Mapping):
        return {"ragas_enabled": raw_ragas.get("enabled")}
    return {
        "ragas_enabled": raw_ragas.get("enabled"),
        **{f"ragas_{key}": value for key, value in scores.items()},
    }


def _answer_from_output(output: Any) -> str:
    if isinstance(output, Mapping):
        return str(output.get("answer", output.get("response", "")))
    return str(getattr(output, "answer", output))


def _is_refusal_or_flagged(output: Any) -> bool:
    answer = _answer_from_output(output)
    if _answer_looks_like_refusal(answer):
        return True
    if not isinstance(output, Mapping):
        return False
    return bool(output.get("low_confidence")) or output.get("grounded") is False


def _answer_looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _confidence_score(output: Any) -> float | None:
    if not isinstance(output, Mapping):
        return None
    value = output.get("confidence_score")
    if value is None and isinstance(output.get("confidence_details"), Mapping):
        value = output["confidence_details"].get("confidence")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _low_confidence(output: Any) -> bool | None:
    if not isinstance(output, Mapping):
        return None
    if "low_confidence" not in output:
        return None
    return bool(output["low_confidence"])


def _chunk_summary(raw_chunk: Any) -> dict[str, Any]:
    if isinstance(raw_chunk, Mapping) and isinstance(raw_chunk.get("chunk"), Mapping):
        raw_chunk = raw_chunk["chunk"]
    if not isinstance(raw_chunk, Mapping):
        return {}
    return {
        "filepath": raw_chunk.get("filepath", raw_chunk.get("path", raw_chunk.get("file", ""))),
        "function_name": raw_chunk.get("function_name", raw_chunk.get("symbol_name", raw_chunk.get("name", ""))),
        "start_line": raw_chunk.get("start_line", 0),
        "end_line": raw_chunk.get("end_line", 0),
    }


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = _csv_fields(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    base = list(CSV_BASE_FIELDS)
    extras = sorted({field for row in rows for field in row if field not in base})
    return base + extras


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _average(values: Sequence[Any] | Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return sum(items) / len(items)


def _optional_path(value: str) -> Path | None:
    if value.lower() in {"", "none", "null"}:
        return None
    return Path(value)


def _print_summary(payload: Mapping[str, Any], *, json_path: Path, csv_path: Path) -> None:
    print("T12 ablation complete")
    for result in payload["conditions"]:
        condition = result["condition"]
        summary = result["summary"]
        print(
            f"{condition['id']} {condition['label']}: "
            f"context_hit={_format_metric(summary.get('context_hit_rate'))}, "
            f"answer_overlap={_format_metric(summary.get('answer_relevance_proxy'))}, "
            f"adversarial_refusal={_format_metric(summary.get('adversarial_refusal_rate'))}"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())

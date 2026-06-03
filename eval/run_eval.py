"""Evaluation harness for repository question-answering pipelines."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DEFAULT_TEST_SET_PATH = Path(__file__).with_name("test_set.json")
REFUSAL_MARKERS = (
    "cannot find",
    "can't find",
    "not in the codebase",
    "not present",
    "no evidence",
    "insufficient context",
    "do not have enough context",
)


@dataclass(frozen=True)
class GroundTruthContext:
    """Expected source location that supports an evaluation answer."""

    filepath: str
    function_name: str | None = None


@dataclass(frozen=True)
class EvalItem:
    """One hand-authored evaluation question and its expected support."""

    id: str
    question: str
    ground_truth_answer: str
    ground_truth_context: list[GroundTruthContext]
    expected_refusal: bool = False


@dataclass(frozen=True)
class PipelineOutput:
    """Normalized answer and retrieved contexts returned by a pipeline."""

    answer: str
    retrieved_contexts: list[GroundTruthContext]
    retrieved_sources: list[str]
    raw_output: Any


def load_test_set(path: str | Path = DEFAULT_TEST_SET_PATH) -> list[EvalItem]:
    """Load and validate the committed manual evaluation set."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation test set must be a JSON list.")

    items: list[EvalItem] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(payload):
        if not isinstance(raw_item, dict):
            raise ValueError(f"Evaluation item {index} must be an object.")

        item_id = _required_str(raw_item, "id", index)
        if item_id in seen_ids:
            raise ValueError(f"Duplicate evaluation item id: {item_id}")
        seen_ids.add(item_id)

        items.append(
            EvalItem(
                id=item_id,
                question=_required_str(raw_item, "question", index),
                ground_truth_answer=_required_str(raw_item, "ground_truth_answer", index),
                ground_truth_context=_parse_contexts(raw_item.get("ground_truth_context", []), index),
                expected_refusal=bool(raw_item.get("expected_refusal", False)),
            )
        )
    return items


def run_eval(
    pipeline_fn: Callable[[str], Any],
    test_set_path: str | Path = DEFAULT_TEST_SET_PATH,
    *,
    use_ragas: bool = False,
) -> dict[str, Any]:
    """Evaluate a question-answering pipeline on the manual test set.

    The pipeline callable receives a question and may return either a plain
    answer string or a mapping with `answer` and `retrieved_chunks`/`contexts`.
    """

    items = load_test_set(test_set_path)
    outputs = [_normalize_pipeline_output(pipeline_fn(item.question)) for item in items]
    records = [_score_item(item, output) for item, output in zip(items, outputs, strict=True)]
    answerable_records = [record for record in records if not record["expected_refusal"]]
    refusal_records = [record for record in records if record["expected_refusal"]]

    metrics: dict[str, Any] = {
        "total_questions": len(items),
        "answerable_questions": len(answerable_records),
        "refusal_questions": len(refusal_records),
        "context_hit_rate": _average(record["context_hit"] for record in answerable_records),
        "context_recall": _average(record["context_recall"] for record in answerable_records),
        "context_precision": _average(record["context_precision"] for record in answerable_records),
        "answer_relevance_proxy": _average(record["answer_token_overlap"] for record in answerable_records),
        "refusal_accuracy": _average(record["refusal_correct"] for record in refusal_records),
        "records": records,
    }
    if use_ragas:
        metrics["ragas"] = _run_ragas_if_available(items, outputs)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the test set or smoke-test the harness with oracle answers."""

    parser = argparse.ArgumentParser(description="Run or validate the manual QA evaluation set.")
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET_PATH), help="Path to eval/test_set.json.")
    parser.add_argument("--oracle", action="store_true", help="Score ground-truth answers as a harness smoke test.")
    parser.add_argument("--ragas", action="store_true", help="Also attempt optional RAGAS scoring.")
    args = parser.parse_args(argv)

    items = load_test_set(args.test_set)
    if not args.oracle:
        print(f"Loaded {len(items)} evaluation questions from {args.test_set}")
        print(f"Refusal questions: {sum(item.expected_refusal for item in items)}")
        return 0

    by_question = {item.question: item for item in items}

    def oracle_pipeline(question: str) -> dict[str, object]:
        item = by_question[question]
        return {
            "answer": item.ground_truth_answer,
            "retrieved_chunks": [asdict(context) for context in item.ground_truth_context],
        }

    metrics = run_eval(oracle_pipeline, args.test_set, use_ragas=args.ragas)
    print(json.dumps(_without_records(metrics), indent=2))
    return 0


def _score_item(item: EvalItem, output: PipelineOutput) -> dict[str, Any]:
    expected_contexts = item.ground_truth_context
    retrieved_contexts = output.retrieved_contexts
    matched_expected = [
        expected for expected in expected_contexts if any(_context_matches(expected, actual) for actual in retrieved_contexts)
    ]
    matched_retrieved = [
        actual for actual in retrieved_contexts if any(_context_matches(expected, actual) for expected in expected_contexts)
    ]

    return {
        "id": item.id,
        "question": item.question,
        "expected_refusal": item.expected_refusal,
        "answer": output.answer,
        "context_hit": bool(matched_expected) if expected_contexts else False,
        "context_recall": len(matched_expected) / len(expected_contexts) if expected_contexts else 0.0,
        "context_precision": len(matched_retrieved) / len(retrieved_contexts) if retrieved_contexts else 0.0,
        "answer_token_overlap": _token_overlap(output.answer, item.ground_truth_answer),
        "refusal_correct": _is_refusal(output.answer) if item.expected_refusal else None,
        "retrieved_contexts": [asdict(context) for context in retrieved_contexts],
        "missing_contexts": [asdict(context) for context in expected_contexts if context not in matched_expected],
    }


def _run_ragas_if_available(items: Sequence[EvalItem], outputs: Sequence[PipelineOutput]) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from langchain_community.llms import Ollama
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as error:
        return {"enabled": False, "reason": f"missing optional dependency: {error.name}"}

    model = os.environ.get("RAGAS_OLLAMA_MODEL", "qwen2.5-coder:7b")
    ragas_llm = LangchainLLMWrapper(Ollama(model=model))
    dataset = Dataset.from_dict(
        {
            "question": [item.question for item in items],
            "answer": [output.answer for output in outputs],
            "contexts": [_ragas_contexts(output) for output in outputs],
            "ground_truth": [item.ground_truth_answer for item in items],
        }
    )

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
            llm=ragas_llm,
        )
    except Exception as error:
        return {"enabled": False, "reason": str(error)}

    return {"enabled": True, "model": model, "scores": _ragas_scores_to_dict(result)}


def _normalize_pipeline_output(output: Any) -> PipelineOutput:
    if isinstance(output, str):
        return PipelineOutput(answer=output, retrieved_contexts=[], retrieved_sources=[], raw_output=output)

    if isinstance(output, dict):
        answer = str(output.get("answer", output.get("response", "")))
        raw_contexts = (
            output.get("retrieved_chunks")
            or output.get("retrieved_contexts")
            or output.get("contexts")
            or output.get("citations")
            or []
        )
        return PipelineOutput(
            answer=answer,
            retrieved_contexts=_normalize_contexts(raw_contexts),
            retrieved_sources=_normalize_context_sources(raw_contexts),
            raw_output=output,
        )

    answer = str(getattr(output, "answer", output))
    raw_contexts = getattr(output, "retrieved_chunks", getattr(output, "contexts", []))
    return PipelineOutput(
        answer=answer,
        retrieved_contexts=_normalize_contexts(raw_contexts),
        retrieved_sources=_normalize_context_sources(raw_contexts),
        raw_output=output,
    )


def _normalize_contexts(raw_contexts: Any) -> list[GroundTruthContext]:
    if isinstance(raw_contexts, (str, bytes)) or raw_contexts is None:
        return []

    contexts: list[GroundTruthContext] = []
    for raw_context in raw_contexts:
        context = _normalize_context(raw_context)
        if context is not None:
            contexts.append(context)
    return contexts


def _normalize_context(raw_context: Any) -> GroundTruthContext | None:
    if isinstance(raw_context, GroundTruthContext):
        return raw_context

    if isinstance(raw_context, dict):
        if "chunk" in raw_context and isinstance(raw_context["chunk"], dict):
            raw_context = raw_context["chunk"]
        filepath = raw_context.get("filepath") or raw_context.get("path") or raw_context.get("file")
        function_name = raw_context.get("function_name") or raw_context.get("symbol_name") or raw_context.get("name")
    else:
        filepath = getattr(raw_context, "filepath", getattr(raw_context, "path", None))
        function_name = getattr(raw_context, "function_name", getattr(raw_context, "symbol_name", None))

    if not filepath:
        return None
    return GroundTruthContext(filepath=str(filepath), function_name=str(function_name) if function_name else None)


def _normalize_context_sources(raw_contexts: Any) -> list[str]:
    if isinstance(raw_contexts, (str, bytes)) or raw_contexts is None:
        return []

    sources: list[str] = []
    for raw_context in raw_contexts:
        source = _context_source(raw_context)
        if source:
            sources.append(source)
    return sources


def _context_source(raw_context: Any) -> str:
    if isinstance(raw_context, dict):
        if "chunk" in raw_context and isinstance(raw_context["chunk"], dict):
            raw_context = raw_context["chunk"]
        source = raw_context.get("source") or raw_context.get("text") or raw_context.get("document")
    else:
        source = getattr(raw_context, "source", getattr(raw_context, "text", ""))
    return str(source) if source else ""


def _parse_contexts(raw_contexts: Any, index: int) -> list[GroundTruthContext]:
    if not isinstance(raw_contexts, list):
        raise ValueError(f"Evaluation item {index} ground_truth_context must be a list.")

    contexts: list[GroundTruthContext] = []
    for raw_context in raw_contexts:
        context = _normalize_context(raw_context)
        if context is None:
            raise ValueError(f"Evaluation item {index} has an invalid ground_truth_context entry.")
        contexts.append(context)
    return contexts


def _required_str(raw_item: dict[str, Any], key: str, index: int) -> str:
    value = raw_item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Evaluation item {index} must include a non-empty string field: {key}")
    return value


def _context_matches(expected: GroundTruthContext, actual: GroundTruthContext) -> bool:
    if expected.filepath != actual.filepath:
        return False
    return expected.function_name is None or expected.function_name == actual.function_name


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _token_overlap(answer: str, ground_truth: str) -> float:
    answer_tokens = set(_tokens(answer))
    truth_tokens = set(_tokens(ground_truth))
    if not truth_tokens:
        return 0.0
    return len(answer_tokens & truth_tokens) / len(truth_tokens)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


def _average(values: Iterable[float | bool | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _format_context(context: GroundTruthContext) -> str:
    if context.function_name:
        return f"{context.filepath}:{context.function_name}"
    return context.filepath


def _ragas_contexts(output: PipelineOutput) -> list[str]:
    if output.retrieved_sources:
        return output.retrieved_sources
    return [_format_context(context) for context in output.retrieved_contexts]


def _without_records(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "records"}


def _ragas_scores_to_dict(result: Any) -> dict[str, float]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return {str(key): float(value) for key, value in frame.mean(numeric_only=True).to_dict().items()}
    if isinstance(result, dict):
        return {str(key): float(value) for key, value in result.items() if isinstance(value, int | float)}
    return {"raw": str(result)}


if __name__ == "__main__":
    raise SystemExit(main())

"""DeepEval GEval judge for existing ablation answer artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.run_eval import DEFAULT_TEST_SET_PATH, EvalItem, load_test_set

DEFAULT_ABLATION_RESULTS_PATH = Path("results/ablation_fastapi.json")
DEFAULT_JSON_OUTPUT_PATH = Path("results/deepeval_fastapi.json")
DEFAULT_CSV_OUTPUT_PATH = Path("results/deepeval_fastapi.csv")
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_BASE_URL = "http://localhost:11434/v1/"
DEFAULT_API_KEY = "dummy"

SCORE_FIELDS = (
    "answer_correctness",
    "faithfulness",
    "refusal_quality",
    "rouge1_f1",
    "rougeL_f1",
    "bleu",
)
CSV_FIELDS = (
    "condition_id",
    "label",
    *SCORE_FIELDS,
    "error_count",
)


@dataclass(frozen=True)
class JudgeInput:
    """Normalized data for one DeepEval LLMTestCase."""

    item_id: str
    question: str
    answer: str
    expected_answer: str
    retrieval_context: list[str]
    expected_refusal: bool


JudgeFn = Callable[[JudgeInput], Mapping[str, Any]]


class DeepEvalJudge:
    """Small adapter around DeepEval so tests can mock judging cleanly."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        configure_local_model: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url
        os.environ.setdefault("LOCAL_MODEL_API_KEY", api_key)
        if configure_local_model:
            configure_deepeval_local_model(model=model, base_url=base_url)

        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase

            try:
                from deepeval.test_case import SingleTurnParams as Params
            except ImportError:
                from deepeval.test_case import LLMTestCaseParams as Params
        except ImportError as error:
            missing = error.name or "deepeval"
            raise RuntimeError(
                f"Missing optional dependency {missing!r}. Install evaluation dependencies with "
                "`pip install -e .[eval]` or install `deepeval` directly."
            ) from error

        self._llm_test_case_cls = LLMTestCase
        self._answer_correctness_metric = GEval(
            name="Answer Correctness",
            criteria=(
                "Determine whether the actual output correctly answers the input question based on the "
                "expected output. Allow paraphrases, equivalent wording, and different ordering when the "
                "meaning is preserved. Penalize contradictions, missing core facts, or invented details."
            ),
            evaluation_params=[
                _param(Params, "INPUT"),
                _param(Params, "ACTUAL_OUTPUT"),
                _param(Params, "EXPECTED_OUTPUT"),
            ],
            threshold=0.5,
        )
        self._faithfulness_metric = GEval(
            name="Faithfulness",
            criteria=(
                "Determine whether every factual claim in the actual output is supported by the retrieval "
                "context. Penalize unsupported implementation details, hallucinated file names, and claims "
                "that are not grounded in the provided context."
            ),
            evaluation_params=[
                _param(Params, "ACTUAL_OUTPUT"),
                _param(Params, "RETRIEVAL_CONTEXT"),
            ],
            threshold=0.5,
        )
        self._refusal_quality_metric = GEval(
            name="Refusal Quality",
            criteria=(
                "For an expected-refusal question, determine whether the actual output clearly refuses or "
                "states that the answer cannot be found in the codebase, without inventing unsupported "
                "implementation details."
            ),
            evaluation_params=[
                _param(Params, "INPUT"),
                _param(Params, "ACTUAL_OUTPUT"),
                _param(Params, "EXPECTED_OUTPUT"),
            ],
            threshold=0.5,
        )

    def __call__(self, judge_input: JudgeInput) -> Mapping[str, Any]:
        test_case = self._llm_test_case_cls(
            input=judge_input.question,
            actual_output=judge_input.answer,
            expected_output=judge_input.expected_answer,
            retrieval_context=judge_input.retrieval_context,
        )

        scores: dict[str, float | None] = {}
        reasons: dict[str, str] = {}

        if judge_input.expected_refusal:
            score, reason = _measure_metric(self._refusal_quality_metric, test_case)
            scores["answer_correctness"] = None
            scores["refusal_quality"] = score
            reasons["refusal_quality"] = reason
        else:
            score, reason = _measure_metric(self._answer_correctness_metric, test_case)
            scores["answer_correctness"] = score
            scores["refusal_quality"] = None
            reasons["answer_correctness"] = reason

        faithfulness, reason = _measure_metric(self._faithfulness_metric, test_case)
        scores["faithfulness"] = faithfulness
        reasons["faithfulness"] = reason

        return {"scores": scores, "reasons": reasons}


def run_deepeval_judge(
    *,
    ablation_results_path: str | Path = DEFAULT_ABLATION_RESULTS_PATH,
    test_set_path: str | Path = Path("eval/test_set_fastapi.json"),
    json_output_path: str | Path = DEFAULT_JSON_OUTPUT_PATH,
    csv_output_path: str | Path = DEFAULT_CSV_OUTPUT_PATH,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    judge_fn: JudgeFn | None = None,
    configure_local_model: bool = True,
) -> dict[str, Any]:
    """Evaluate all ablation conditions with GEval and lexical baselines."""

    test_items = {item.id: item for item in load_test_set(test_set_path)}
    ablation_payload = _read_json(ablation_results_path)
    judge = judge_fn or DeepEvalJudge(
        model=model,
        base_url=base_url,
        api_key=api_key,
        configure_local_model=configure_local_model,
    )

    condition_payloads: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for condition_result in _conditions_from_ablation(ablation_payload):
        condition = dict(condition_result.get("condition", {}))
        records = [
            _judge_record(record, test_items=test_items, judge_fn=judge)
            for record in _records_from_condition(condition_result)
        ]
        summary = summarize_records(records)
        condition_payloads.append({"condition": condition, "summary": summary, "records": records})
        csv_rows.append(_csv_row(condition=condition, summary=summary))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge": {
            "tool": "deepeval",
            "model": model,
            "base_url": base_url,
        },
        "source": {
            "ablation_results_path": str(ablation_results_path),
            "test_set_path": str(test_set_path),
        },
        "conditions": condition_payloads,
    }
    _write_json(payload, json_output_path)
    _write_csv(csv_rows, csv_output_path)
    return payload


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average per-question scores into a per-condition summary."""

    summary = {field: _average(_score_value(record, field) for record in records) for field in SCORE_FIELDS}
    summary["error_count"] = sum(1 for record in records if record.get("error"))
    return summary


def build_judge_input(record: Mapping[str, Any], item: EvalItem) -> JudgeInput:
    """Build one LLMTestCase-compatible input from ablation output and test data."""

    return JudgeInput(
        item_id=str(record.get("id", item.id)),
        question=str(record.get("question", item.question)),
        answer=str(record.get("answer", "")),
        expected_answer=item.ground_truth_answer,
        retrieval_context=_retrieval_context(record.get("retrieved_contexts", [])),
        expected_refusal=bool(record.get("expected_refusal", item.expected_refusal)),
    )


def lexical_baselines(answer: str, expected_answer: str) -> dict[str, float]:
    """Compute lightweight lexical baselines for comparison only."""

    answer_tokens = _tokens(answer)
    expected_tokens = _tokens(expected_answer)
    return {
        "rouge1_f1": _rouge1_f1(answer_tokens, expected_tokens),
        "rougeL_f1": _rouge_l_f1(answer_tokens, expected_tokens),
        "bleu": _bleu1(answer_tokens, expected_tokens),
    }


def configure_deepeval_local_model(*, model: str, base_url: str) -> None:
    """Configure DeepEval's local-model endpoint using the documented CLI."""

    try:
        subprocess.run(
            [
                _deepeval_cli(),
                "set-local-model",
                f"--model={model}",
                f"--base-url={base_url}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "DeepEval CLI is unavailable. Install evaluation dependencies with "
            "`pip install -e .[eval]` or `pip install deepeval`."
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(f"Failed to configure DeepEval local model: {detail}") from error


def _deepeval_cli() -> str:
    sibling = Path(sys.executable).with_name("deepeval")
    if sibling.exists():
        return str(sibling)
    return shutil.which("deepeval") or "deepeval"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Judge existing ablation answers with DeepEval GEval and write JSON/CSV artifacts."
    )
    parser.add_argument(
        "--ablation-results",
        default=str(DEFAULT_ABLATION_RESULTS_PATH),
        help="Path to results/ablation_fastapi.json.",
    )
    parser.add_argument(
        "--test-set",
        default=str(DEFAULT_TEST_SET_PATH.with_name("test_set_fastapi.json")),
        help="Path to eval/test_set_fastapi.json.",
    )
    parser.add_argument(
        "--json-output",
        default=str(DEFAULT_JSON_OUTPUT_PATH),
        help="Destination DeepEval JSON artifact.",
    )
    parser.add_argument(
        "--csv-output",
        default=str(DEFAULT_CSV_OUTPUT_PATH),
        help="Destination DeepEval CSV summary artifact.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local judge model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible local judge base URL.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Value for LOCAL_MODEL_API_KEY.")
    parser.add_argument(
        "--skip-local-model-config",
        action="store_true",
        help="Skip `deepeval set-local-model` when it has already been configured.",
    )
    args = parser.parse_args(argv)

    payload = run_deepeval_judge(
        ablation_results_path=args.ablation_results,
        test_set_path=args.test_set,
        json_output_path=args.json_output,
        csv_output_path=args.csv_output,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        configure_local_model=not args.skip_local_model_config,
    )
    _print_summary(payload, json_path=Path(args.json_output), csv_path=Path(args.csv_output))
    return 0


def _judge_record(
    record: Mapping[str, Any],
    *,
    test_items: Mapping[str, EvalItem],
    judge_fn: JudgeFn,
) -> dict[str, Any]:
    item_id = str(record.get("id", ""))
    if item_id not in test_items:
        raise ValueError(f"Ablation record id {item_id!r} is missing from the test set.")

    item = test_items[item_id]
    judge_input = build_judge_input(record, item)
    lexical_scores = (
        {field: None for field in ("rouge1_f1", "rougeL_f1", "bleu")}
        if judge_input.expected_refusal
        else lexical_baselines(judge_input.answer, judge_input.expected_answer)
    )
    scores: dict[str, Any] = {
        "answer_correctness": None,
        "faithfulness": None,
        "refusal_quality": None,
        **lexical_scores,
    }
    reasons: dict[str, str] = {}
    error: str | None = None

    try:
        judge_result = judge_fn(judge_input)
        raw_scores = judge_result.get("scores", judge_result)
        if isinstance(raw_scores, Mapping):
            for field in ("answer_correctness", "faithfulness", "refusal_quality"):
                scores[field] = _coerce_score(raw_scores.get(field))
        raw_reasons = judge_result.get("reasons")
        if isinstance(raw_reasons, Mapping):
            reasons = {str(key): str(value) for key, value in raw_reasons.items() if value is not None}
    except Exception as exc:
        error = str(exc)

    return {
        "id": judge_input.item_id,
        "question": judge_input.question,
        "expected_refusal": judge_input.expected_refusal,
        "answer": judge_input.answer,
        "ground_truth_answer": judge_input.expected_answer,
        "retrieval_context": judge_input.retrieval_context,
        "scores": scores,
        "reasons": reasons,
        "error": error,
    }


def _measure_metric(metric: Any, test_case: Any) -> tuple[float | None, str]:
    metric.measure(test_case)
    score = _coerce_score(getattr(metric, "score", None))
    reason = str(getattr(metric, "reason", "") or "")
    return score, reason


def _param(params: Any, name: str) -> Any:
    value = getattr(params, name, None)
    if value is not None:
        return value
    lowered = name.lower()
    value = getattr(params, lowered, None)
    if value is not None:
        return value
    raise RuntimeError(f"DeepEval parameter {name} is unavailable in this installed version.")


def _conditions_from_ablation(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    conditions = payload.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("Ablation results must include a `conditions` list.")
    return [condition for condition in conditions if isinstance(condition, Mapping)]


def _records_from_condition(condition_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    metrics = condition_result.get("metrics")
    if not isinstance(metrics, Mapping):
        return []
    records = metrics.get("records", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _retrieval_context(raw_contexts: Any) -> list[str]:
    if not isinstance(raw_contexts, Sequence) or isinstance(raw_contexts, (str, bytes)):
        return []

    context: list[str] = []
    for raw_context in raw_contexts:
        if isinstance(raw_context, Mapping):
            filepath = raw_context.get("filepath") or raw_context.get("path") or raw_context.get("file")
            function_name = raw_context.get("function_name") or raw_context.get("symbol_name") or raw_context.get("name")
            source = raw_context.get("source") or raw_context.get("text") or raw_context.get("document")
            location = _format_location(filepath, function_name)
            if source and location:
                context.append(f"{location}\n{source}")
            elif source:
                context.append(str(source))
            elif location:
                context.append(location)
        elif raw_context is not None:
            context.append(str(raw_context))
    return context


def _format_location(filepath: Any, function_name: Any) -> str:
    if not filepath:
        return ""
    location = str(filepath)
    if function_name:
        location = f"{location}::{function_name}"
    return location


def _score_value(record: Mapping[str, Any], field: str) -> float | None:
    scores = record.get("scores")
    if not isinstance(scores, Mapping):
        return None
    return _coerce_score(scores.get(field))


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return max(0.0, min(1.0, score))


def _rouge1_f1(answer_tokens: Sequence[str], expected_tokens: Sequence[str]) -> float:
    if not answer_tokens or not expected_tokens:
        return 0.0
    answer_counts = Counter(answer_tokens)
    expected_counts = Counter(expected_tokens)
    overlap = sum((answer_counts & expected_counts).values())
    return _f1(overlap / len(answer_tokens), overlap / len(expected_tokens))


def _rouge_l_f1(answer_tokens: Sequence[str], expected_tokens: Sequence[str]) -> float:
    if not answer_tokens or not expected_tokens:
        return 0.0
    lcs = _lcs_length(answer_tokens, expected_tokens)
    return _f1(lcs / len(answer_tokens), lcs / len(expected_tokens))


def _bleu1(answer_tokens: Sequence[str], expected_tokens: Sequence[str]) -> float:
    if not answer_tokens or not expected_tokens:
        return 0.0
    answer_counts = Counter(answer_tokens)
    expected_counts = Counter(expected_tokens)
    overlap = sum((answer_counts & expected_counts).values())
    precision = overlap / len(answer_tokens)
    brevity_penalty = 1.0 if len(answer_tokens) > len(expected_tokens) else math.exp(1 - len(expected_tokens) / len(answer_tokens))
    return brevity_penalty * precision


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


def _average(values: Sequence[Any] | Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return sum(items) / len(items)


def _csv_row(*, condition: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": condition.get("id", ""),
        "label": condition.get("label", ""),
        **{field: summary.get(field) for field in SCORE_FIELDS},
        "error_count": summary.get("error_count", 0),
    }


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _print_summary(payload: Mapping[str, Any], *, json_path: Path, csv_path: Path) -> None:
    print("DeepEval judge complete")
    for condition_result in payload["conditions"]:
        condition = condition_result["condition"]
        summary = condition_result["summary"]
        print(
            f"{condition.get('id')} {condition.get('label')}: "
            f"correctness={_format_metric(summary.get('answer_correctness'))}, "
            f"faithfulness={_format_metric(summary.get('faithfulness'))}, "
            f"refusal={_format_metric(summary.get('refusal_quality'))}, "
            f"errors={summary.get('error_count', 0)}"
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

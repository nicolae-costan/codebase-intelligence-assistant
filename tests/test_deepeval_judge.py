import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval.ablation import DEFAULT_CONDITIONS
from eval.deepeval_judge import (
    JudgeInput,
    build_judge_input,
    run_deepeval_judge,
    summarize_records,
)
from eval.run_eval import load_test_set

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_deepeval_judge_writes_condition_artifacts(tmp_path: Path) -> None:
    test_set_path = _write_test_set(tmp_path)
    ablation_path = _write_ablation_results(tmp_path)
    json_path = tmp_path / "results" / "deepeval.json"
    csv_path = tmp_path / "results" / "deepeval.csv"
    calls: list[JudgeInput] = []

    def fake_judge(judge_input: JudgeInput) -> dict[str, Any]:
        calls.append(judge_input)
        if judge_input.item_id == "judge-failure":
            raise RuntimeError("judge model failed")
        if judge_input.expected_refusal:
            return {
                "scores": {
                    "answer_correctness": None,
                    "faithfulness": 0.9,
                    "refusal_quality": 1.0,
                },
                "reasons": {"refusal_quality": "The answer refuses clearly."},
            }
        return {
            "scores": {
                "answer_correctness": 0.8,
                "faithfulness": 0.6,
                "refusal_quality": None,
            },
            "reasons": {"answer_correctness": "The answer matches the expected behavior."},
        }

    payload = run_deepeval_judge(
        ablation_results_path=ablation_path,
        test_set_path=test_set_path,
        json_output_path=json_path,
        csv_output_path=csv_path,
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434/v1/",
        judge_fn=fake_judge,
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert [result["condition"]["id"] for result in payload["conditions"]] == ["A", "B", "C", "D"]
    assert len(calls) == 4 * 3
    assert calls[0].retrieval_context == ["src/app.py::run"]
    assert calls[1].expected_refusal is True

    first_condition = payload["conditions"][0]
    assert first_condition["summary"]["answer_correctness"] == 0.8
    assert first_condition["summary"]["faithfulness"] == 0.75
    assert first_condition["summary"]["refusal_quality"] == 1.0
    assert first_condition["summary"]["error_count"] == 1
    assert first_condition["records"][1]["scores"]["refusal_quality"] == 1.0
    assert first_condition["records"][1]["scores"]["rouge1_f1"] is None
    assert first_condition["records"][2]["error"] == "judge model failed"

    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["judge"]["tool"] == "deepeval"
    assert persisted["judge"]["model"] == "qwen2.5-coder:7b"
    assert persisted["conditions"][0]["records"][0]["ground_truth_answer"] == "run returns the app status."

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 4
    assert rows[0]["condition_id"] == "A"
    assert rows[0]["label"] == "Dense only, single-pass"
    assert rows[0]["answer_correctness"] == "0.800000"
    assert rows[0]["faithfulness"] == "0.750000"
    assert rows[0]["refusal_quality"] == "1.000000"
    assert rows[0]["error_count"] == "1"


def test_build_judge_input_uses_ground_truth_and_context_summaries(tmp_path: Path) -> None:
    test_set_path = _write_test_set(tmp_path)
    item = load_test_set(test_set_path)[0]
    judge_input = build_judge_input(
        {
            "id": "answerable",
            "question": "Where is run defined?",
            "answer": "run returns the status",
            "retrieved_contexts": [
                {
                    "filepath": "src/app.py",
                    "function_name": "run",
                    "source": "def run(): return status",
                }
            ],
        },
        item,
    )

    assert judge_input.question == "Where is run defined?"
    assert judge_input.expected_answer == "run returns the app status."
    assert judge_input.retrieval_context == ["src/app.py::run\ndef run(): return status"]
    assert judge_input.expected_refusal is False


def test_summarize_records_averages_scores_and_counts_errors() -> None:
    records = [
        {
            "scores": {
                "answer_correctness": 0.2,
                "faithfulness": 0.5,
                "refusal_quality": None,
                "rouge1_f1": 0.4,
                "rougeL_f1": 0.3,
                "bleu": 0.2,
            },
            "error": None,
        },
        {
            "scores": {
                "answer_correctness": None,
                "faithfulness": 0.9,
                "refusal_quality": 1.0,
                "rouge1_f1": None,
                "rougeL_f1": None,
                "bleu": None,
            },
            "error": "judge failed",
        },
    ]

    summary = summarize_records(records)

    assert summary["answer_correctness"] == 0.2
    assert summary["faithfulness"] == 0.7
    assert summary["refusal_quality"] == 1.0
    assert summary["rouge1_f1"] == 0.4
    assert summary["rougeL_f1"] == 0.3
    assert summary["bleu"] == 0.2
    assert summary["error_count"] == 1


def test_deepeval_judge_cli_help_is_available_without_ollama_or_deepeval() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "eval.deepeval_judge", "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Judge existing ablation answers with DeepEval GEval" in result.stdout
    assert "--ablation-results" in result.stdout
    assert "--skip-local-model-config" in result.stdout


def _write_test_set(tmp_path: Path) -> Path:
    path = tmp_path / "test_set.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "answerable",
                    "question": "Where is run defined?",
                    "ground_truth_answer": "run returns the app status.",
                    "ground_truth_context": [{"filepath": "src/app.py", "function_name": "run"}],
                    "expected_refusal": False,
                },
                {
                    "id": "missing",
                    "question": "Where is the billing dashboard?",
                    "ground_truth_answer": "I cannot find this in the codebase.",
                    "ground_truth_context": [],
                    "expected_refusal": True,
                },
                {
                    "id": "judge-failure",
                    "question": "Where is the failing branch?",
                    "ground_truth_answer": "The failing branch is in src/fail.py.",
                    "ground_truth_context": [{"filepath": "src/fail.py", "function_name": "fail"}],
                    "expected_refusal": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_ablation_results(tmp_path: Path) -> Path:
    path = tmp_path / "ablation.json"
    path.write_text(
        json.dumps(
            {
                "conditions": [
                    {
                        "condition": {
                            "id": condition.id,
                            "label": condition.label,
                            "retrieval": condition.retrieval,
                            "retrieval_passes": condition.retrieval_passes,
                            "mode": condition.mode,
                        },
                        "metrics": {
                            "records": [
                                {
                                    "id": "answerable",
                                    "question": "Where is run defined?",
                                    "expected_refusal": False,
                                    "answer": "run returns the app status.",
                                    "retrieved_contexts": [
                                        {"filepath": "src/app.py", "function_name": "run"}
                                    ],
                                },
                                {
                                    "id": "missing",
                                    "question": "Where is the billing dashboard?",
                                    "expected_refusal": True,
                                    "answer": "I cannot find this in the codebase.",
                                    "retrieved_contexts": [],
                                },
                                {
                                    "id": "judge-failure",
                                    "question": "Where is the failing branch?",
                                    "expected_refusal": False,
                                    "answer": "The failing branch is in src/fail.py.",
                                    "retrieved_contexts": [
                                        {"filepath": "src/fail.py", "function_name": "fail"}
                                    ],
                                },
                            ]
                        },
                    }
                    for condition in DEFAULT_CONDITIONS
                ]
            }
        ),
        encoding="utf-8",
    )
    return path

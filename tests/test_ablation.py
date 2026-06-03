import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval.ablation import DEFAULT_CONDITIONS, run_ablation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_ablation_writes_four_condition_artifacts(tmp_path: Path) -> None:
    test_set = _write_test_set(
        tmp_path,
        [
            {
                "id": "answerable",
                "question": "Where is run defined?",
                "ground_truth_answer": "run is defined in src/app.py.",
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
        ],
    )
    json_path = tmp_path / "results" / "ablation.json"
    csv_path = tmp_path / "results" / "ablation.csv"
    calls: list[dict[str, Any]] = []

    def retriever_factory(retrieval: str):
        def retriever(query: str, top_k: int):
            return [
                {
                    "filepath": "src/app.py",
                    "function_name": "run",
                    "start_line": 1,
                    "end_line": 3,
                    "source": f"retrieved by {retrieval}",
                }
            ]

        return retriever

    def pipeline_runner(question: str, **kwargs):
        chunks = kwargs["retriever"](question, kwargs["top_k"])
        calls.append(
            {
                "question": question,
                "mode": kwargs["mode"],
                "confidence": kwargs["confidence"],
                "confidence_threshold": kwargs["confidence_threshold"],
            }
        )
        if "billing" in question or "OAuth" in question:
            return {
                "answer": "I cannot find this in the codebase.",
                "retrieved_chunks": [],
                "confidence_score": 0.2,
                "confidence_level": "low",
                "low_confidence": True,
                "grounded": True,
                "ungrounded_claims": [],
            }
        return {
            "answer": "run is defined in src/app.py.",
            "retrieved_chunks": chunks,
            "confidence_score": 0.9,
            "confidence_level": "high",
            "low_confidence": False,
            "grounded": True,
            "ungrounded_claims": [],
        }

    payload = run_ablation(
        test_set_path=test_set,
        output_json_path=json_path,
        output_csv_path=csv_path,
        top_k=3,
        confidence_threshold=0.75,
        pipeline_log_path=None,
        confidence_log_path=None,
        adversarial_queries=["Where is OAuth token rotation implemented?"],
        retriever_factory=retriever_factory,
        pipeline_runner=pipeline_runner,
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert [result["condition"]["id"] for result in payload["conditions"]] == ["A", "B", "C", "D"]
    assert [condition.id for condition in DEFAULT_CONDITIONS] == ["A", "B", "C", "D"]
    assert len(calls) == 4 * 3
    assert {call["mode"] for call in calls} == {"single", "iterative"}
    assert all(call["confidence"] is True for call in calls)
    assert all(call["confidence_threshold"] == 0.75 for call in calls)

    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["top_k"] == 3
    assert persisted["confidence_enabled"] is True
    assert persisted["conditions"][0]["summary"]["context_hit_rate"] == 1.0
    assert persisted["conditions"][0]["adversarial"]["refusal_rate"] == 1.0

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 4
    assert rows[0]["condition_id"] == "A"
    assert rows[0]["retrieval"] == "dense"
    assert rows[1]["retrieval"] == "hybrid"
    assert rows[2]["retrieval_passes"] == "iterative (2-pass)"
    assert rows[0]["adversarial_refusal_rate"] == "1.000000"


def test_adversarial_flag_counts_low_confidence_as_success(tmp_path: Path) -> None:
    test_set = _write_test_set(
        tmp_path,
        [
            {
                "id": "simple",
                "question": "Where is run defined?",
                "ground_truth_answer": "run is defined in src/app.py.",
                "ground_truth_context": [{"filepath": "src/app.py", "function_name": "run"}],
                "expected_refusal": False,
            }
        ],
    )

    def retriever_factory(_retrieval: str):
        return lambda _query, _top_k: [{"filepath": "src/app.py", "function_name": "run"}]

    def pipeline_runner(question: str, **kwargs):
        if "invented" in question:
            return {
                "answer": "The invented API is handled by run.",
                "retrieved_chunks": [{"filepath": "src/app.py", "function_name": "run"}],
                "confidence_score": 0.31,
                "low_confidence": True,
                "grounded": True,
                "ungrounded_claims": [],
            }
        return {
            "answer": "run is defined in src/app.py.",
            "retrieved_chunks": [{"filepath": "src/app.py", "function_name": "run"}],
            "confidence_score": 0.88,
            "low_confidence": False,
            "grounded": True,
            "ungrounded_claims": [],
        }

    payload = run_ablation(
        test_set_path=test_set,
        output_json_path=tmp_path / "ablation.json",
        output_csv_path=tmp_path / "ablation.csv",
        pipeline_log_path=None,
        confidence_log_path=None,
        adversarial_queries=["Where is the invented API implemented?"],
        retriever_factory=retriever_factory,
        pipeline_runner=pipeline_runner,
    )

    assert payload["conditions"][0]["adversarial"]["records"][0]["correctly_refused_or_flagged"] is True
    assert payload["conditions"][0]["summary"]["adversarial_refusal_rate"] == 1.0


def test_ablation_cli_help_is_available_without_indexes_or_ollama() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "eval.ablation", "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run T12 ablation experiments" in result.stdout
    assert "--ragas" in result.stdout
    assert "--adversarial-query" in result.stdout


def _write_test_set(tmp_path: Path, payload: list[dict[str, object]]) -> Path:
    path = tmp_path / "test_set.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

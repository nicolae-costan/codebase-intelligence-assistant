import json
from pathlib import Path

from eval.run_eval import DEFAULT_TEST_SET_PATH, _normalize_pipeline_output, load_test_set, run_eval


def test_committed_test_set_loads_with_required_fields() -> None:
    items = load_test_set(DEFAULT_TEST_SET_PATH)

    assert 15 <= len(items) <= 20
    assert all(item.id for item in items)
    assert all(item.question for item in items)
    assert all(item.ground_truth_answer for item in items)
    assert any(item.expected_refusal for item in items)
    assert any(not item.expected_refusal and item.ground_truth_context for item in items)


def test_run_eval_scores_context_retrieval_with_mock_pipeline(tmp_path: Path) -> None:
    test_set = _write_test_set(
        tmp_path,
        [
            {
                "id": "answerable",
                "question": "Where is run defined?",
                "ground_truth_answer": "run is defined in src/app.py.",
                "ground_truth_context": [{"filepath": "src/app.py", "function_name": "run"}],
                "expected_refusal": False,
            }
        ],
    )

    def pipeline(question: str) -> dict[str, object]:
        return {
            "answer": "run is defined in src/app.py.",
            "retrieved_chunks": [{"filepath": "src/app.py", "function_name": "run"}],
        }

    metrics = run_eval(pipeline, test_set)

    assert metrics["total_questions"] == 1
    assert metrics["context_hit_rate"] == 1.0
    assert metrics["context_recall"] == 1.0
    assert metrics["context_precision"] == 1.0


def test_run_eval_scores_refusal_questions(tmp_path: Path) -> None:
    test_set = _write_test_set(
        tmp_path,
        [
            {
                "id": "missing",
                "question": "Where is the dense index?",
                "ground_truth_answer": "I cannot find this in the codebase.",
                "ground_truth_context": [],
                "expected_refusal": True,
            }
        ],
    )

    metrics = run_eval(lambda question: "I cannot find this in the codebase.", test_set)

    assert metrics["refusal_questions"] == 1
    assert metrics["refusal_accuracy"] == 1.0


def test_run_eval_accepts_nested_chunk_outputs(tmp_path: Path) -> None:
    test_set = _write_test_set(
        tmp_path,
        [
            {
                "id": "nested",
                "question": "Where is Chunk?",
                "ground_truth_answer": "Chunk is in src/schema.py.",
                "ground_truth_context": [{"filepath": "src/schema.py", "function_name": "Chunk"}],
                "expected_refusal": False,
            }
        ],
    )

    def pipeline(question: str) -> dict[str, object]:
        return {
            "answer": "Chunk is in src/schema.py.",
            "retrieved_chunks": [{"chunk": {"filepath": "src/schema.py", "function_name": "Chunk"}}],
        }

    metrics = run_eval(pipeline, test_set)

    assert metrics["records"][0]["retrieved_contexts"] == [{"filepath": "src/schema.py", "function_name": "Chunk"}]
    assert metrics["context_recall"] == 1.0


def test_pipeline_output_preserves_retrieved_sources_for_ragas() -> None:
    output = _normalize_pipeline_output(
        {
            "answer": "run returns 1.",
            "retrieved_chunks": [{"filepath": "src/app.py", "function_name": "run", "source": "def run():\n    return 1\n"}],
        }
    )

    assert output.retrieved_contexts[0].filepath == "src/app.py"
    assert output.retrieved_sources == ["def run():\n    return 1\n"]


def test_ragas_path_degrades_when_optional_dependencies_are_missing(tmp_path: Path) -> None:
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

    metrics = run_eval(lambda question: "run is defined in src/app.py.", test_set, use_ragas=True)

    assert "ragas" in metrics
    assert "enabled" in metrics["ragas"]


def _write_test_set(tmp_path: Path, payload: list[dict[str, object]]) -> Path:
    path = tmp_path / "test_set.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

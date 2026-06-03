import json
from pathlib import Path

from src.confidence import DEFAULT_TEMPERATURES, estimate_confidence
from src.generator import REFUSAL_ANSWER
from src.schema import Chunk


class RecordingGenerator:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[tuple[list[dict[str, object]], str, float]] = []

    def __call__(self, context_chunks: list[dict[str, object]], query: str, *, temperature: float) -> str:
        self.calls.append((list(context_chunks), query, temperature))
        return self.answers[len(self.calls) - 1]


def test_estimate_confidence_generates_three_temperature_variants() -> None:
    generator = RecordingGenerator(["run returns 1. [1]", "run returns 1. [1]", "run returns 1. [1]"])

    result = estimate_confidence(
        "What does run do?",
        [_chunk_dict()],
        generator_fn=generator,
        semantic_similarity_fn=lambda _answers: [1.0, 1.0, 1.0],
        log_path=None,
    )

    assert [call[2] for call in generator.calls] == list(DEFAULT_TEMPERATURES)
    assert all(call[0] == [_chunk_dict()] for call in generator.calls)
    assert all(call[1] == "What does run do?" for call in generator.calls)
    assert result.answer == "run returns 1. [1]"
    assert result.confidence == 1.0
    assert result.level == "high"
    assert result.should_refuse is False
    assert result.warning is None


def test_low_agreement_answers_are_flagged() -> None:
    generator = RecordingGenerator(
        [
            "run returns 1. [1]",
            "clone_repository writes files. [1]",
            "Chunk stores metadata. [1]",
        ]
    )

    result = estimate_confidence(
        "What does run do?",
        [_chunk_dict()],
        generator_fn=generator,
        semantic_similarity_fn=lambda _answers: [0.0, 0.1, 0.0],
        threshold=0.72,
        log_path=None,
    )

    assert result.answer == "run returns 1. [1]"
    assert result.confidence < 0.72
    assert result.level == "low"
    assert result.should_refuse is True
    assert result.warning is not None
    assert len(result.pair_scores) == 3


def test_empty_context_returns_refusal_without_generation() -> None:
    generator = RecordingGenerator(["should not be used"])

    result = estimate_confidence(
        "Unknown?",
        [],
        generator_fn=generator,
        semantic_similarity_fn=lambda _answers: [1.0, 1.0, 1.0],
        log_path=None,
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.should_refuse is True
    assert result.warning is not None
    assert generator.calls == []


def test_confidence_logging_writes_jsonl_record(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "confidence.jsonl"
    generator = RecordingGenerator(["run returns 1. [1]", "run returns 1. [1]", "run returns 1. [1]"])

    estimate_confidence(
        "What does run do?",
        [_chunk_dict()],
        generator_fn=generator,
        semantic_similarity_fn=lambda _answers: [1.0, 1.0, 1.0],
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["timestamp"]
    assert record["query"] == "What does run do?"
    assert record["temperatures"] == list(DEFAULT_TEMPERATURES)
    assert record["answer"] == "run returns 1. [1]"
    assert record["confidence"] == 1.0
    assert record["level"] == "high"
    assert len(record["pair_scores"]) == 3


def test_log_path_none_disables_confidence_file_writes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    generator = RecordingGenerator(["run returns 1. [1]", "run returns 1. [1]", "run returns 1. [1]"])

    estimate_confidence(
        "What does run do?",
        [_chunk_dict()],
        generator_fn=generator,
        semantic_similarity_fn=lambda _answers: [1.0, 1.0, 1.0],
        log_path=None,
    )

    assert not (tmp_path / "results").exists()


def _chunk_dict() -> dict[str, object]:
    return Chunk(
        id="run",
        filepath="src/app.py",
        function_name="run",
        start_line=10,
        end_line=12,
        docstring="Run the app.",
        source="def run():\n    return 1\n",
        language="python",
    ).to_dict()

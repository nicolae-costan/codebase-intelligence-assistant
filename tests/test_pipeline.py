import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.generator import REFUSAL_ANSWER
from src.pipeline import iterative_rag, single_pass_rag
from src.schema import Chunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeRetriever:
    def __init__(self, responses: list[list[Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, top_k: int) -> list[Any]:
        self.calls.append((query, top_k))
        return self.responses[len(self.calls) - 1]


class FakeGenerator:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[tuple[list[dict[str, object]], str]] = []

    def __call__(self, context_chunks: list[dict[str, object]], query: str) -> str:
        self.calls.append((list(context_chunks), query))
        return self.answers[len(self.calls) - 1]


class FakeConfidenceEstimator:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.result


def test_single_mode_calls_retriever_and_generator_once() -> None:
    retriever = FakeRetriever([[{"chunk": _chunk("first").to_dict(), "score": 0.9}]])
    generator = FakeGenerator(["final answer"])

    result = iterative_rag(
        "Where is run defined?",
        mode="single",
        top_k=3,
        retriever=retriever,
        generator_fn=generator,
        log_path=None,
    )

    assert retriever.calls == [("Where is run defined?", 3)]
    assert len(generator.calls) == 1
    assert generator.calls[0] == ([_chunk_dict("first")], "Where is run defined?")
    assert result["answer"] == "final answer"
    assert result["retrieved_chunks"] == [_chunk_dict("first")]
    assert result["mode"] == "single"
    assert "partial_answer" not in result
    assert "confidence_score" not in result


def test_single_pass_rag_wraps_single_mode() -> None:
    retriever = FakeRetriever([[_chunk("only")]])
    generator = FakeGenerator(["single answer"])

    result = single_pass_rag("What is this?", retriever=retriever, generator_fn=generator, log_path=None)

    assert result["mode"] == "single"
    assert result["answer"] == "single answer"
    assert retriever.calls == [("What is this?", 7)]


def test_iterative_mode_runs_two_retrieval_and_generation_passes() -> None:
    retriever = FakeRetriever([[_chunk("first")], [{"chunk": _chunk_dict("second"), "rank": 1}]])
    generator = FakeGenerator(["partial answer", "final answer"])

    result = iterative_rag(
        "How does run work?",
        mode="iterative",
        top_k=5,
        retriever=retriever,
        generator_fn=generator,
        log_path=None,
    )

    assert len(retriever.calls) == 2
    assert retriever.calls[0] == ("How does run work?", 5)
    second_query, second_top_k = retriever.calls[1]
    assert second_top_k == 5
    assert "How does run work?" in second_query
    assert "Partial answer:\npartial answer" in second_query
    assert generator.calls[0] == ([_chunk_dict("first")], "How does run work?")
    assert generator.calls[1] == ([_chunk_dict("second")], "How does run work?")
    assert result["answer"] == "final answer"
    assert result["partial_answer"] == "partial answer"
    assert result["retrieved_chunks"] == [_chunk_dict("second")]
    assert result["mode"] == "iterative"


def test_final_answer_uses_second_pass_chunks_not_first_pass_chunks() -> None:
    retriever = FakeRetriever([[_chunk("stale")], [_chunk("fresh")]])

    def generator(context_chunks: list[dict[str, object]], query: str) -> str:
        return f"answer from {context_chunks[0]['id']}" if context_chunks else "empty"

    result = iterative_rag(
        "Which chunk wins?",
        mode="iterative",
        retriever=retriever,
        generator_fn=generator,
        log_path=None,
    )

    assert result["answer"] == "answer from fresh"
    assert result["retrieved_chunks"] == [_chunk_dict("fresh")]


def test_pipeline_expands_linked_chunks_with_deduplication() -> None:
    retriever = FakeRetriever(
        [
            [
                {
                    "chunk": _chunk_dict("primary"),
                    "linked_chunks": [_chunk_dict("neighbor"), _chunk_dict("primary")],
                }
            ]
        ]
    )
    generator = FakeGenerator(["answer"])

    result = iterative_rag("How does the big symbol work?", retriever=retriever, generator_fn=generator, log_path=None)

    assert generator.calls[0] == ([_chunk_dict("primary"), _chunk_dict("neighbor")], "How does the big symbol work?")
    assert result["retrieved_chunks"] == [_chunk_dict("primary"), _chunk_dict("neighbor")]


def test_pipeline_preserves_retrieval_debug_for_primary_chunks() -> None:
    retriever = FakeRetriever(
        [
            [
                {
                    "chunk": _chunk_dict("primary"),
                    "linked_chunks": [_chunk_dict("neighbor")],
                    "retrieval_debug": {"dense_rank": 1, "bm25_rank": 3, "rrf_score": 0.031},
                }
            ]
        ]
    )
    generator = FakeGenerator(["answer"])

    result = iterative_rag("How does the big symbol work?", retriever=retriever, generator_fn=generator, log_path=None)

    assert result["retrieved_chunks"][0]["retrieval_debug"] == {
        "dense_rank": 1,
        "bm25_rank": 3,
        "rrf_score": 0.031,
    }
    assert "retrieval_debug" not in result["retrieved_chunks"][1]


def test_iterative_mode_skips_second_pass_when_first_pass_refuses() -> None:
    retriever = FakeRetriever([[_chunk("first")]])
    generator = FakeGenerator([REFUSAL_ANSWER])

    result = iterative_rag(
        "Unknown?",
        mode="iterative",
        retriever=retriever,
        generator_fn=generator,
        log_path=None,
    )

    assert retriever.calls == [("Unknown?", 7)]
    assert generator.calls == [([_chunk_dict("first")], "Unknown?")]
    assert result["answer"] == REFUSAL_ANSWER
    assert result["trace"]["pass_2"] == {"skipped": "pass_1_refused"}


def test_confidence_uses_final_iterative_chunks_and_adds_metadata() -> None:
    retriever = FakeRetriever([[_chunk("stale")], [_chunk("fresh")]])
    generator = FakeGenerator(["partial answer", "final answer"])
    confidence_estimator = FakeConfidenceEstimator(
        {
            "answer": "confidence answer from fresh",
            "confidence": 0.91,
            "level": "high",
            "should_refuse": False,
            "raw_answers": ["confidence answer from fresh"],
        }
    )

    result = iterative_rag(
        "Which chunk wins?",
        mode="iterative",
        retriever=retriever,
        generator_fn=generator,
        confidence=True,
        confidence_threshold=0.8,
        confidence_log_path=None,
        confidence_estimator=confidence_estimator,
        log_path=None,
    )

    assert result["answer"] == "confidence answer from fresh"
    assert result["retrieved_chunks"] == [_chunk_dict("fresh")]
    assert result["confidence_score"] == 0.91
    assert result["confidence_level"] == "high"
    assert result["low_confidence"] is False
    [call] = confidence_estimator.calls
    assert call["query"] == "Which chunk wins?"
    assert call["context_chunks"] == [_chunk_dict("fresh")]
    assert call["generator_fn"] is generator
    assert call["threshold"] == 0.8
    assert call["log_path"] is None


def test_low_confidence_is_surfaced_without_changing_retrieved_chunks() -> None:
    retriever = FakeRetriever([[_chunk("first")]])
    generator = FakeGenerator(["normal answer"])
    confidence_estimator = FakeConfidenceEstimator(
        {
            "answer": "uncertain answer",
            "confidence": 0.41,
            "level": "low",
            "should_refuse": True,
            "warning": "Low confidence: answer agreement 0.41 is below the configured threshold 0.72.",
            "raw_answers": ["uncertain answer"],
        }
    )

    result = iterative_rag(
        "What happens?",
        retriever=retriever,
        generator_fn=generator,
        confidence=True,
        confidence_estimator=confidence_estimator,
        log_path=None,
        confidence_log_path=None,
    )

    assert result["answer"] == REFUSAL_ANSWER
    assert result["confidence_candidate_answer"] == "uncertain answer"
    assert result["retrieved_chunks"] == [_chunk_dict("first")]
    assert result["confidence_score"] == 0.41
    assert result["confidence_level"] == "low"
    assert result["low_confidence"] is True
    assert result["confidence_warning"] == "Low confidence: answer agreement 0.41 is below the configured threshold 0.72."
    assert result["confidence_details"]["should_refuse"] is True
    assert result["confidence_details"]["answer"] == "uncertain answer"


def test_empty_retrieval_uses_generator_refusal_behavior() -> None:
    retriever = FakeRetriever([[]])

    result = iterative_rag("Unknown?", retriever=retriever, log_path=None)

    assert result["answer"] == REFUSAL_ANSWER
    assert result["retrieved_chunks"] == []


def test_jsonl_logging_writes_one_valid_record(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "iterative.jsonl"
    retriever = FakeRetriever([[_chunk("first")], [{"chunk": _chunk_dict("second")}]])
    generator = FakeGenerator(["partial", "final"])

    iterative_rag(
        "How?",
        mode="iterative",
        top_k=2,
        retriever=retriever,
        generator_fn=generator,
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["timestamp"]
    assert record["mode"] == "iterative"
    assert record["top_k"] == 2
    assert record["partial_answer"] == "partial"
    assert record["answer"] == "final"
    assert record["pass_1"]["retrieved_chunks"][0]["id"] == "first"
    assert record["pass_2"]["retrieved_chunks"][0]["filepath"] == "src/second.py"
    assert record["pass_1"]["retrieval_ms"] >= 0
    assert record["pass_2"]["generation_ms"] >= 0
    assert record["latency_ms"] >= 0


def test_log_path_none_disables_file_writes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    retriever = FakeRetriever([[_chunk("first")]])
    generator = FakeGenerator(["final"])

    iterative_rag("No log", retriever=retriever, generator_fn=generator, log_path=None)

    assert not (tmp_path / "results").exists()


def test_cli_help_is_available_without_retriever_or_ollama() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run single-pass or iterative RAG" in result.stdout
    assert "--mode" in result.stdout
    assert "--confidence" in result.stdout
    assert "--confidence-threshold" in result.stdout


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        filepath=f"src/{chunk_id}.py",
        function_name=f"{chunk_id}_fn",
        start_line=1,
        end_line=2,
        docstring=f"{chunk_id} docs",
        source=f"def {chunk_id}_fn():\n    return {chunk_id!r}\n",
        language="python",
    )


def _chunk_dict(chunk_id: str) -> dict[str, object]:
    return _chunk(chunk_id).to_dict()

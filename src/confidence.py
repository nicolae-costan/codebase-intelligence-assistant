"""HonestCoder-style confidence estimation for grounded codebase answers."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from src.generator import REFUSAL_ANSWER, generate

DEFAULT_CONFIDENCE_THRESHOLD = 0.72
DEFAULT_CONFIDENCE_LOG_PATH = Path("results/confidence_log.jsonl")
DEFAULT_TEMPERATURES = (0.0, 0.5, 0.9)
LEXICAL_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SIMILARITY_MODEL: Any | None = None

GeneratorFn = Callable[..., str]
SemanticSimilarityFn = Callable[[Sequence[str]], Sequence[float]]


@dataclass(frozen=True)
class ConfidenceResult:
    """Structured confidence decision for one generated answer."""

    answer: str
    confidence: float
    level: str
    should_refuse: bool
    warning: str | None
    raw_answers: list[str]
    semantic_similarity: float
    lexical_similarity: float
    pair_scores: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def estimate_confidence(
    query: str,
    context_chunks: Sequence[Mapping[str, object]],
    *,
    generator_fn: GeneratorFn | None = None,
    temps: Sequence[float] = DEFAULT_TEMPERATURES,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    log_path: str | Path | None = DEFAULT_CONFIDENCE_LOG_PATH,
    semantic_similarity_fn: SemanticSimilarityFn | None = None,
) -> ConfidenceResult:
    """Generate answer variants, score agreement, and return confidence metadata."""

    if not temps:
        raise ValueError("temps must include at least one temperature.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    if not context_chunks:
        result = ConfidenceResult(
            answer=REFUSAL_ANSWER,
            confidence=0.0,
            level="low",
            should_refuse=True,
            warning="Low confidence: no retrieved context was available.",
            raw_answers=[REFUSAL_ANSWER],
            semantic_similarity=0.0,
            lexical_similarity=0.0,
            pair_scores=[],
        )
        _append_confidence_log(result, query=query, threshold=threshold, temps=temps, log_path=log_path)
        return result

    answer_generator = generator_fn or generate
    answers = [
        _generate_with_temperature(answer_generator, context_chunks=context_chunks, query=query, temperature=temperature)
        for temperature in temps
    ]
    pair_scores = _pairwise_scores(answers, semantic_similarity_fn=semantic_similarity_fn)
    lexical_similarity = _average(float(score["lexical_similarity"]) for score in pair_scores)
    semantic_similarity = _average(float(score["semantic_similarity"]) for score in pair_scores)
    confidence = _average(float(score["combined_similarity"]) for score in pair_scores)
    selected_answer = answers[0]
    level = _confidence_level(confidence, threshold)
    should_refuse = selected_answer == REFUSAL_ANSWER or confidence < threshold
    result = ConfidenceResult(
        answer=selected_answer,
        confidence=confidence,
        level=level,
        should_refuse=should_refuse,
        warning=_confidence_warning(
            selected_answer=selected_answer,
            confidence=confidence,
            threshold=threshold,
            should_refuse=should_refuse,
        ),
        raw_answers=answers,
        semantic_similarity=semantic_similarity,
        lexical_similarity=lexical_similarity,
        pair_scores=pair_scores,
    )
    _append_confidence_log(result, query=query, threshold=threshold, temps=temps, log_path=log_path)
    return result


def _generate_with_temperature(
    generator_fn: GeneratorFn,
    *,
    context_chunks: Sequence[Mapping[str, object]],
    query: str,
    temperature: float,
) -> str:
    return generator_fn(context_chunks, query, temperature=temperature)


def _pairwise_scores(
    answers: Sequence[str],
    *,
    semantic_similarity_fn: SemanticSimilarityFn | None,
) -> list[dict[str, object]]:
    pairs = list(combinations(range(len(answers)), 2))
    if not pairs:
        return []

    semantic_scores = list(
        semantic_similarity_fn(answers) if semantic_similarity_fn is not None else _semantic_pairwise_similarities(answers)
    )
    if len(semantic_scores) != len(pairs):
        raise ValueError("semantic_similarity_fn must return one score for each answer pair.")

    scores: list[dict[str, object]] = []
    for pair_index, ((left, right), semantic_similarity) in enumerate(zip(pairs, semantic_scores, strict=True)):
        lexical_similarity = _lexical_similarity(answers[left], answers[right])
        combined_similarity = (LEXICAL_WEIGHT * lexical_similarity) + (SEMANTIC_WEIGHT * semantic_similarity)
        scores.append(
            {
                "pair": [left, right],
                "lexical_similarity": _round_score(lexical_similarity),
                "semantic_similarity": _round_score(semantic_similarity),
                "combined_similarity": _round_score(combined_similarity),
                "answer_left": answers[left],
                "answer_right": answers[right],
                "pair_index": pair_index,
            }
        )
    return scores


def _semantic_pairwise_similarities(answers: Sequence[str]) -> list[float]:
    model = _load_similarity_model()
    embeddings = model.encode(list(answers))
    return [_cosine_similarity(embeddings[left], embeddings[right]) for left, right in combinations(range(len(answers)), 2)]


def _load_similarity_model() -> Any:
    global _SIMILARITY_MODEL
    if _SIMILARITY_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Confidence estimation requires the `sentence-transformers` package. "
                "Install project dependencies with `python -m pip install -e .`."
            ) from exc
        _SIMILARITY_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SIMILARITY_MODEL


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(left.lower()))
    right_tokens = set(_TOKEN_RE.findall(right.lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _cosine_similarity(left_embedding: Any, right_embedding: Any) -> float:
    left_values = [float(value) for value in left_embedding]
    right_values = [float(value) for value in right_embedding]
    dot_product = sum(left * right for left, right in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot_product / (left_norm * right_norm)))


def _confidence_level(confidence: float, threshold: float) -> str:
    if confidence >= threshold:
        return "high"
    if confidence >= max(0.0, threshold - 0.15):
        return "medium"
    return "low"


def _confidence_warning(*, selected_answer: str, confidence: float, threshold: float, should_refuse: bool) -> str | None:
    if not should_refuse:
        return None
    if selected_answer == REFUSAL_ANSWER:
        return "Low confidence: the generator could not ground an answer in the retrieved context."
    return f"Low confidence: answer agreement {confidence:.2f} is below the configured threshold {threshold:.2f}."


def _append_confidence_log(
    result: ConfidenceResult,
    *,
    query: str,
    threshold: float,
    temps: Sequence[float],
    log_path: str | Path | None,
) -> None:
    if log_path is None:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "threshold": threshold,
        "temperatures": list(temps),
        **result.to_dict(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _average(values: Sequence[float] | Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return _round_score(sum(items) / len(items))


def _round_score(value: float) -> float:
    return round(float(value), 6)

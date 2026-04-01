"""Think@n aggregation utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Callable, Dict, Tuple

from deep_think.types import CandidateSample, SequenceAnalysis


def rank_candidates_by_dtr(
    candidates: Sequence[CandidateSample],
) -> list[CandidateSample]:
    """Sort candidates by descending DTR score."""

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.rank_score,
            candidate.analysis.dtr,
            -candidate.generated_tokens,
        ),
        reverse=True,
    )


def select_candidates(
    candidates: Sequence[CandidateSample],
    top_k: int,
) -> list[CandidateSample]:
    """Select the top-ranked candidates."""

    return rank_candidates_by_dtr(candidates)[:top_k]


def majority_vote_text(
    candidates: Sequence[CandidateSample],
    extractor: Callable[[SequenceAnalysis], str] | None = None,
) -> Tuple[str, Dict[str, int]]:
    """Aggregate candidates by text or a caller-supplied answer extractor."""

    keyed_candidates: list[tuple[str, CandidateSample]] = []
    for candidate in candidates:
        if extractor is None:
            key = candidate.analysis.text.strip()
        else:
            key = extractor(candidate.analysis).strip()
        if key:
            keyed_candidates.append((key, candidate))

    counter = Counter(key for key, _ in keyed_candidates)
    if not counter:
        return "", {}

    winning_key, _ = max(
        counter.items(), key=lambda item: (item[1], len(item[0]))
    )
    winning_candidate = next(
        candidate
        for key, candidate in keyed_candidates
        if key == winning_key
    )
    return winning_candidate.analysis.text.strip(), dict(counter)


def estimated_think_at_n_cost(candidates: Sequence[CandidateSample]) -> int:
    """Estimate token cost from the tokens actually generated for all candidates."""

    return sum(candidate.generated_tokens for candidate in candidates)

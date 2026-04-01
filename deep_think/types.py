"""Typed result containers used by the library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence


@dataclass(frozen=True)
class TokenTrace:
    """Per-token DTR diagnostics."""

    token_id: int
    token_text: str
    settling_depth: int
    is_deep_thinking: bool
    entropy: float
    mean_jsd_to_final: float
    layer_jsds_to_final: Sequence[float] = field(default_factory=tuple)


@dataclass(frozen=True)
class SequenceAnalysis:
    """Sequence-level output plus DTR statistics."""

    prompt: str
    text: str
    token_ids: Sequence[int]
    token_texts: Sequence[str]
    dtr: float
    mean_jsd: float
    mean_entropy: float
    settling_depths: Sequence[int]
    deep_token_mask: Sequence[bool]
    token_traces: Sequence[TokenTrace]


@dataclass(frozen=True)
class CandidateSample:
    """One sampled candidate used in Think@n."""

    rank_score: float
    analysis: SequenceAnalysis
    prefix_analysis: SequenceAnalysis
    generated_tokens: int
    was_pruned: bool = False


@dataclass(frozen=True)
class ThinkAtNResult:
    """Output of Think@n sampling and aggregation."""

    final_text: str
    selected_candidates: Sequence[CandidateSample]
    all_candidates: Sequence[CandidateSample]
    vote_distribution: Dict[str, int]
    estimated_cost_tokens: int

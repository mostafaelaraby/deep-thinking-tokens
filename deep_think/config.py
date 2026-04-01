"""Configuration objects for DTR analysis and Think@n generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from deep_think.types import SequenceAnalysis


@dataclass(frozen=True)
class DTRConfig:
    """Configuration for deep-thinking ratio computation."""

    settling_threshold: float = 0.5
    depth_fraction: float = 0.85
    top_k_logits: Optional[int] = None
    every_k_layers: int = 1
    include_embedding_layer: bool = False
    apply_final_norm: bool = True
    exclude_special_tokens: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for one-shot generation and analysis."""

    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None


@dataclass(frozen=True)
class ThinkAtNConfig:
    """Configuration for Think@n repeated sampling and selection."""

    num_samples: int = 8
    prefix_tokens: int = 50
    select_top_k: Optional[int] = None
    do_majority_vote: bool = True
    rank_by_prefix_dtr: bool = False
    early_prune_by_prefix: bool = False
    vote_text_extractor: Optional[Callable[[SequenceAnalysis], str]] = None

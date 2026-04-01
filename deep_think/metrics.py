"""Pure metric computations for DTR."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import List, Optional, Tuple

import torch


def shannon_entropy(probabilities: torch.Tensor) -> float:
    """Compute entropy from a probability vector."""

    probs = probabilities.clamp_min(1e-12)
    return float(-(probs * probs.log()).sum().item())


def jsd_to_final(
    final_probs: torch.Tensor,
    layer_probs: torch.Tensor,
    top_k: Optional[int] = None,
) -> float:
    """Compute Jensen-Shannon divergence to the final-layer distribution."""

    p = final_probs.float()
    q = layer_probs.float()
    if top_k is not None and top_k < p.shape[-1]:
        indices = torch.topk(p + q, k=top_k).indices
        p = p.index_select(0, indices)
        q = q.index_select(0, indices)
        p = p / p.sum().clamp_min(1e-12)
        q = q / q.sum().clamp_min(1e-12)

    m = 0.5 * (p + q)
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    m = m.clamp_min(1e-12)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float(0.5 * (kl_pm + kl_qm))


def settling_depth(
    jsds_to_final: Sequence[float],
    settling_threshold: float,
    layer_indices: Optional[Sequence[int]] = None,
) -> int:
    """Return the first layer where the running minimum crosses the threshold."""

    if not jsds_to_final:
        return 0

    if layer_indices is None:
        resolved_indices = list(range(len(jsds_to_final)))
    else:
        if len(layer_indices) != len(jsds_to_final):
            raise ValueError("layer_indices must align with jsds_to_final.")
        resolved_indices = [int(index) for index in layer_indices]

    running_min = float("inf")
    for index, jsd in enumerate(jsds_to_final):
        running_min = min(running_min, float(jsd))
        if running_min <= settling_threshold:
            return resolved_indices[index]
    return resolved_indices[-1]


def deep_token_threshold(num_layers: int, depth_fraction: float) -> int:
    """Compute the late-regime threshold on the paper's 1-based layer axis."""

    return max(1, int(ceil(depth_fraction * num_layers)))


def classify_token(
    jsds_to_final: Sequence[float],
    settling_threshold_value: float,
    depth_fraction: float,
    layer_indices: Optional[Sequence[int]] = None,
    total_layers: Optional[int] = None,
) -> Tuple[int, bool]:
    """Return settling depth and deep-thinking label for one token."""

    depth = settling_depth(
        jsds_to_final,
        settling_threshold_value,
        layer_indices=layer_indices,
    )
    if total_layers is None:
        if layer_indices is None:
            total_layers = len(jsds_to_final)
        else:
            total_layers = max(int(index) for index in layer_indices)
    threshold = deep_token_threshold(total_layers, depth_fraction)
    return depth, depth >= threshold


def aggregate_sequence_metrics(
    settling_depths: Sequence[int],
    deep_token_mask: Sequence[bool],
    entropies: Sequence[float],
    mean_jsds: Sequence[float],
) -> Tuple[float, float, float]:
    """Aggregate token-level metrics into sequence-level scores."""

    token_count = max(len(settling_depths), 1)
    dtr = sum(1 for is_deep in deep_token_mask if is_deep) / token_count
    mean_jsd = sum(mean_jsds) / max(len(mean_jsds), 1)
    mean_entropy = sum(entropies) / max(len(entropies), 1)
    return dtr, mean_jsd, mean_entropy


def validate_probability_sequence(layer_probabilities: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Validate that there are enough per-layer distributions to compute DTR."""

    if len(layer_probabilities) < 2:
        raise ValueError("DTR requires at least two selected layers.")
    return list(layer_probabilities)

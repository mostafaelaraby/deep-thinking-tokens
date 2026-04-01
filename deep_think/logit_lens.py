"""Helpers for projecting hidden states to vocabulary probabilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, List, Optional

import torch

from deep_think.config import DTRConfig


def get_transformer_norm(
    model: torch.nn.Module,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Return the final normalization module when present."""

    for attr_path in (
        ("model", "norm"),
        ("model", "decoder", "final_layer_norm"),
        ("transformer", "ln_f"),
        ("transformer", "norm"),
    ):
        current = model
        found = True
        for attr in attr_path:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found and callable(current):
            return current
    return None


def get_lm_head(model: torch.nn.Module) -> torch.nn.Module:
    """Return the LM head used to map hidden states to logits."""

    for attr in ("lm_head", "embed_out", "output_projection"):
        if hasattr(model, attr):
            head = getattr(model, attr)
            if callable(head):
                return head
    raise AttributeError("Could not locate an LM head on the supplied model.")


def select_hidden_state_indices(
    num_hidden_states: int,
    config: DTRConfig,
) -> List[int]:
    """Choose which hidden states to use for DTR computation."""

    start = 0 if config.include_embedding_layer else 1
    step = max(config.every_k_layers, 1)
    indices = list(range(start, num_hidden_states, step))
    last = num_hidden_states - 1
    if last not in indices:
        indices.append(last)
    if len(indices) < 2:
        indices = list(range(start, num_hidden_states))
    return indices


def hidden_state_to_probs(
    hidden_state: torch.Tensor,
    lm_head: torch.nn.Module,
    final_norm: Optional[Callable[[torch.Tensor], torch.Tensor]],
    apply_final_norm: bool = True,
) -> torch.Tensor:
    """Project a hidden state into vocabulary probabilities.

    By default this applies the model's final normalization before the
    unembedding projection, matching the logit-lens setup used by the paper.
    """

    vector = hidden_state
    if apply_final_norm and final_norm is not None:
        vector = final_norm(vector.unsqueeze(0)).squeeze(0)
    logits = lm_head(vector)
    return torch.softmax(logits.float(), dim=-1)


def token_layer_probabilities(
    hidden_states: Sequence[torch.Tensor],
    token_position: int,
    selected_indices: Sequence[int],
    lm_head: torch.nn.Module,
    final_norm: Optional[Callable[[torch.Tensor], torch.Tensor]],
    apply_final_norm: bool = True,
) -> List[torch.Tensor]:
    """Return probability distributions for one token across selected layers."""

    return [
        hidden_state_to_probs(
            hidden_states[index][0, token_position],
            lm_head,
            final_norm,
            apply_final_norm=apply_final_norm,
        )
        for index in selected_indices
    ]

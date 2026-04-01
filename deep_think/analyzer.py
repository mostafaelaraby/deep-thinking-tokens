"""Sequence analyzer that computes DTR from generated tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Set

import torch

from deep_think.config import DTRConfig
from deep_think.logit_lens import (
    get_lm_head,
    get_transformer_norm,
    hidden_state_to_probs,
    select_hidden_state_indices,
    token_layer_probabilities,
)
from deep_think.metrics import (
    aggregate_sequence_metrics,
    classify_token,
    jsd_to_final,
    shannon_entropy,
    validate_probability_sequence,
)
from deep_think.types import SequenceAnalysis, TokenTrace
from deep_think.utils import decode_text


class SequenceAnalyzer:
    """Analyze generated tokens using hidden-state trajectories."""

    def __init__(
        self, model: torch.nn.Module, tokenizer: object, config: DTRConfig
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.lm_head = get_lm_head(model)
        self.final_norm = get_transformer_norm(model)

    def _excluded_token_ids(self) -> Set[int]:
        """Return tokenizer special-token ids that should not affect DTR aggregates."""

        if not self.config.exclude_special_tokens:
            return set()

        token_ids: Set[int] = set()
        for attr_name in (
            "all_special_ids",
            "eos_token_id",
            "bos_token_id",
            "pad_token_id",
            "sep_token_id",
            "cls_token_id",
            "mask_token_id",
        ):
            value = getattr(self.tokenizer, attr_name, None)
            if value is None:
                continue
            if isinstance(value, int):
                token_ids.add(int(value))
            elif isinstance(value, (list, tuple, set)):
                token_ids.update(
                    int(token_id)
                    for token_id in value
                    if token_id is not None
                )
        return token_ids

    def _empty_analysis(self, prompt: str) -> SequenceAnalysis:
        return SequenceAnalysis(
            prompt=prompt,
            text="",
            token_ids=[],
            token_texts=[],
            dtr=0.0,
            mean_jsd=0.0,
            mean_entropy=0.0,
            settling_depths=[],
            deep_token_mask=[],
            token_traces=[],
        )

    def next_token_distribution(
        self,
        hidden_states: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return the next-token distribution from the final hidden state."""

        token_position = int(hidden_states[-1].shape[1]) - 1
        return hidden_state_to_probs(
            hidden_states[-1][0, token_position],
            self.lm_head,
            self.final_norm,
            apply_final_norm=self.config.apply_final_norm,
        )

    def _trace_from_hidden_states(
        self,
        token_id: int,
        token_text: str,
        hidden_states: Sequence[torch.Tensor],
        selected_indices: Sequence[int],
    ) -> TokenTrace:
        """Build the paper's per-token trajectory from one decoding step.

        Each decoding step gives us the hidden state at every layer for the current
        prefix. We project those layer states to vocabulary probabilities, compare
        them to the final-layer distribution with JSD, and then compute the token's
        settling depth.
        """

        token_position = int(hidden_states[0].shape[1]) - 1
        if token_position < 0:
            raise ValueError(
                "Prompt must contain at least one token before generation starts."
            )

        layer_probabilities = token_layer_probabilities(
            hidden_states,
            token_position,
            selected_indices,
            self.lm_head,
            self.final_norm,
            apply_final_norm=self.config.apply_final_norm,
        )
        layer_probabilities = validate_probability_sequence(layer_probabilities)
        final_probs = layer_probabilities[-1]
        jsds = [
            jsd_to_final(final_probs, layer_probs, top_k=self.config.top_k_logits)
            for layer_probs in layer_probabilities
        ]
        mean_jsd = sum(jsds[:-1]) / max(len(jsds) - 1, 1)
        entropy = shannon_entropy(final_probs)
        settling_depth, is_deep = classify_token(
            jsds,
            settling_threshold_value=self.config.settling_threshold,
            depth_fraction=self.config.depth_fraction,
            layer_indices=selected_indices,
            total_layers=max(len(hidden_states) - 1, 1),
        )
        return TokenTrace(
            token_id=int(token_id),
            token_text=token_text,
            settling_depth=settling_depth,
            is_deep_thinking=is_deep,
            entropy=entropy,
            mean_jsd_to_final=mean_jsd,
            layer_jsds_to_final=tuple(jsds),
        )

    def analyze_hidden_state_steps(
        self,
        prompt: str,
        token_ids: Sequence[int],
        hidden_state_steps: Sequence[Sequence[torch.Tensor]],
    ) -> SequenceAnalysis:
        """Compute DTR from the hidden states captured during token-by-token decoding."""

        if len(token_ids) != len(hidden_state_steps):
            raise ValueError(
                "Expected one hidden-state snapshot per generated token."
            )
        if not token_ids:
            return self._empty_analysis(prompt)

        # We use the same layer subset for every token in the sequence.
        selected_indices = select_hidden_state_indices(
            len(hidden_state_steps[0]), self.config
        )
        token_texts = [
            decode_text(self.tokenizer, [token_id]) for token_id in token_ids
        ]
        excluded_token_ids = self._excluded_token_ids()

        traces: List[TokenTrace] = []
        aggregate_entropies: List[float] = []
        aggregate_mean_jsds: List[float] = []
        aggregate_deep_token_mask: List[bool] = []
        aggregate_settling_depths: List[int] = []

        for token_id, token_text, hidden_states in zip(
            token_ids, token_texts, hidden_state_steps
        ):
            # One trace corresponds to one generated token at one decoding step.
            trace = self._trace_from_hidden_states(
                token_id=int(token_id),
                token_text=token_text,
                hidden_states=hidden_states,
                selected_indices=selected_indices,
            )
            traces.append(trace)

            # DTR aggregates skip special tokens such as EOS by default, but we
            # still keep those traces for inspection.
            if int(token_id) in excluded_token_ids:
                continue
            aggregate_settling_depths.append(trace.settling_depth)
            aggregate_deep_token_mask.append(trace.is_deep_thinking)
            aggregate_entropies.append(trace.entropy)
            aggregate_mean_jsds.append(trace.mean_jsd_to_final)

        dtr, sequence_mean_jsd, mean_entropy = aggregate_sequence_metrics(
            aggregate_settling_depths,
            aggregate_deep_token_mask,
            aggregate_entropies,
            aggregate_mean_jsds,
        )
        return SequenceAnalysis(
            prompt=prompt,
            text=decode_text(self.tokenizer, list(token_ids)),
            token_ids=list(token_ids),
            token_texts=token_texts,
            dtr=dtr,
            mean_jsd=sequence_mean_jsd,
            mean_entropy=mean_entropy,
            settling_depths=[trace.settling_depth for trace in traces],
            deep_token_mask=[trace.is_deep_thinking for trace in traces],
            token_traces=traces,
        )

    @torch.no_grad()
    def analyze(
        self,
        prompt: str,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> SequenceAnalysis:
        """Replay a generated continuation token by token and compute DTR."""

        if generated_ids.ndim != 2 or generated_ids.shape[0] != 1:
            raise ValueError(
                "Expected generated_ids to have shape [1, seq_len]."
            )

        token_ids = generated_ids[0].tolist()
        if not token_ids:
            return self._empty_analysis(prompt)

        running_ids = prompt_ids.to(self.model.device)
        attention_mask = torch.ones_like(running_ids)
        hidden_state_steps: List[Sequence[torch.Tensor]] = []
        replayed_token_ids: List[int] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        for token_id in token_ids:
            outputs = self.model(
                running_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_state_steps.append(tuple(outputs.hidden_states))
            replayed_token_ids.append(int(token_id))

            next_token = torch.tensor(
                [[int(token_id)]],
                device=running_ids.device,
                dtype=running_ids.dtype,
            )
            running_ids = torch.cat([running_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones_like(next_token, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
            if eos_token_id is not None and int(token_id) == int(eos_token_id):
                break

        return self.analyze_hidden_state_steps(
            prompt=prompt,
            token_ids=replayed_token_ids,
            hidden_state_steps=hidden_state_steps,
        )

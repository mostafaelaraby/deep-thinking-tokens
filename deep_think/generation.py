"""High-level Hugging Face integration API."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from deep_think.aggregation import (
    estimated_think_at_n_cost,
    majority_vote_text,
    select_candidates,
)
from deep_think.analyzer import SequenceAnalyzer
from deep_think.config import DTRConfig, GenerationConfig, ThinkAtNConfig
from deep_think.types import CandidateSample, SequenceAnalysis, ThinkAtNResult
from deep_think.utils import build_prompt, pick_top_k


class DeepThinker:
    """Wrapper that adds DTR analysis and Think@n sampling to a causal LM."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: object,
        dtr_config: Optional[DTRConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dtr_config = dtr_config or DTRConfig()
        self.analyzer = SequenceAnalyzer(
            model=model,
            tokenizer=tokenizer,
            config=self.dtr_config,
        )

    @property
    def device(self) -> torch.device:
        """Return the model device."""

        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def _encode_prompt(self, prompt: str) -> tuple[str, torch.Tensor, torch.Tensor]:
        formatted_prompt = build_prompt(self.tokenizer, prompt)
        encoded = self.tokenizer(formatted_prompt, return_tensors="pt")
        prompt_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        return formatted_prompt, prompt_ids, attention_mask

    def _normalize_sampling_distribution(
        self,
        probabilities: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        distribution = probabilities.float().clamp_min(1e-12)
        temperature = generation_config.temperature
        if temperature <= 0:
            raise ValueError("GenerationConfig.temperature must be positive.")
        if temperature != 1.0:
            distribution = distribution.pow(1.0 / temperature)
            distribution = distribution / distribution.sum().clamp_min(1e-12)

        top_p = generation_config.top_p
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(distribution, descending=True)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_remove = cumulative_probs > top_p
            if sorted_remove.numel() > 1:
                sorted_remove[1:] = sorted_remove[:-1].clone()
            sorted_remove[0] = False
            filtered = sorted_probs.masked_fill(sorted_remove, 0.0)
            filtered = filtered / filtered.sum().clamp_min(1e-12)
            distribution = torch.zeros_like(distribution)
            distribution.scatter_(0, sorted_indices, filtered)

        return distribution / distribution.sum().clamp_min(1e-12)

    def _pick_next_token_id(
        self,
        probabilities: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> int:
        if not generation_config.do_sample:
            return int(torch.argmax(probabilities).item())

        distribution = self._normalize_sampling_distribution(
            probabilities,
            generation_config,
        )
        return int(torch.multinomial(distribution, num_samples=1).item())

    def _decode_steps(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig,
        max_new_tokens_override: Optional[int] = None,
        forced_generated_ids: Optional[torch.Tensor] = None,
    ) -> tuple[list[int], list[Sequence[torch.Tensor]]]:
        """Decode one sequence token by token and keep the hidden-state trace."""

        max_new_tokens = max_new_tokens_override or generation_config.max_new_tokens
        generated_token_ids: list[int] = []
        hidden_state_steps: list[Sequence[torch.Tensor]] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        if forced_generated_ids is not None:
            if forced_generated_ids.ndim == 1:
                forced_generated_ids = forced_generated_ids.unsqueeze(0)
            if forced_generated_ids.ndim != 2 or forced_generated_ids.shape[0] != 1:
                raise ValueError(
                    "Expected forced_generated_ids to have shape [1, seq_len]."
                )
            forced_token_ids = forced_generated_ids[0].tolist()
            max_new_tokens = min(max_new_tokens, len(forced_token_ids))
        else:
            forced_token_ids = []

        prompt_len = prompt_ids.shape[1]
        total_len = prompt_len + max_new_tokens
        device = prompt_ids.device

        running_ids = torch.zeros(
            (1, total_len), device=device, dtype=prompt_ids.dtype
        )
        running_ids[0, :prompt_len] = prompt_ids[0]

        running_mask = torch.zeros(
            (1, total_len), device=device, dtype=attention_mask.dtype
        )
        running_mask[0, :prompt_len] = attention_mask[0]

        seq_end = prompt_len

        for step in range(max_new_tokens):
            outputs = self.model(
                running_ids[:, :seq_end],
                attention_mask=running_mask[:, :seq_end],
                output_hidden_states=True,
            )
            hidden_state_steps.append(tuple(outputs.hidden_states))
            next_token_probs = self.analyzer.next_token_distribution(
                outputs.hidden_states
            )
            if forced_generated_ids is None:
                next_token_id = self._pick_next_token_id(
                    next_token_probs,
                    generation_config,
                )
            else:
                next_token_id = int(forced_token_ids[step])

            generated_token_ids.append(next_token_id)
            running_ids[0, seq_end] = next_token_id
            running_mask[0, seq_end] = 1
            seq_end += 1

            if eos_token_id is not None and next_token_id == int(eos_token_id):
                break

        return generated_token_ids, hidden_state_steps

    def _analysis_from_steps(
        self,
        formatted_prompt: str,
        token_ids: Sequence[int],
        hidden_state_steps: Sequence[Sequence[torch.Tensor]],
    ) -> SequenceAnalysis:
        return self.analyzer.analyze_hidden_state_steps(
            prompt=formatted_prompt,
            token_ids=token_ids,
            hidden_state_steps=hidden_state_steps,
        )

    def _sampling_config(
        self,
        generation_config: GenerationConfig,
        num_return_sequences: int,
    ) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=generation_config.pad_token_id,
        )

    def _sample_candidates_full_generation(
        self,
        formatted_prompt: str,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        think_config: ThinkAtNConfig,
        generation_config: GenerationConfig,
    ) -> list[CandidateSample]:
        candidates: list[CandidateSample] = []
        for _ in range(think_config.num_samples):
            token_ids, hidden_state_steps = self._decode_steps(
                prompt_ids,
                attention_mask,
                generation_config,
            )
            analysis = self._analysis_from_steps(
                formatted_prompt,
                token_ids,
                hidden_state_steps,
            )
            prefix_len = min(len(token_ids), think_config.prefix_tokens)
            prefix_analysis = self._analysis_from_steps(
                formatted_prompt,
                token_ids[:prefix_len],
                hidden_state_steps[:prefix_len],
            )
            rank_score = prefix_analysis.dtr if think_config.rank_by_prefix_dtr else analysis.dtr
            candidates.append(
                CandidateSample(
                    rank_score=rank_score,
                    analysis=analysis,
                    prefix_analysis=prefix_analysis,
                    generated_tokens=len(token_ids),
                    was_pruned=False,
                )
            )
        return candidates

    def _sample_candidates_with_early_pruning(
        self,
        formatted_prompt: str,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        think_config: ThinkAtNConfig,
        generation_config: GenerationConfig,
    ) -> list[CandidateSample]:
        prefix_budget = min(
            think_config.prefix_tokens,
            generation_config.max_new_tokens,
        )
        if prefix_budget <= 0 or prefix_budget >= generation_config.max_new_tokens:
            return self._sample_candidates_full_generation(
                formatted_prompt,
                prompt_ids,
                attention_mask,
                think_config,
                generation_config,
            )

        top_k = pick_top_k(think_config.num_samples, think_config.select_top_k)
        prefix_candidates: list[CandidateSample] = []
        prefix_traces: list[tuple[list[int], list[Sequence[torch.Tensor]]]] = []

        for _ in range(think_config.num_samples):
            prefix_token_ids, prefix_hidden_state_steps = self._decode_steps(
                prompt_ids,
                attention_mask,
                generation_config,
                max_new_tokens_override=prefix_budget,
            )
            prefix_analysis = self._analysis_from_steps(
                formatted_prompt,
                prefix_token_ids,
                prefix_hidden_state_steps,
            )
            prefix_traces.append((prefix_token_ids, prefix_hidden_state_steps))
            prefix_candidates.append(
                CandidateSample(
                    rank_score=prefix_analysis.dtr,
                    analysis=prefix_analysis,
                    prefix_analysis=prefix_analysis,
                    generated_tokens=len(prefix_token_ids),
                    was_pruned=True,
                )
            )

        selected_prefixes = select_candidates(prefix_candidates, top_k=top_k)
        selected_index_set = {id(candidate) for candidate in selected_prefixes}
        remaining_tokens = generation_config.max_new_tokens - prefix_budget
        completed_by_index: dict[int, CandidateSample] = {}

        for candidate_index, prefix_candidate in enumerate(prefix_candidates):
            if id(prefix_candidate) not in selected_index_set:
                continue

            prefix_token_ids, prefix_hidden_state_steps = prefix_traces[candidate_index]
            prefix_tensor = torch.tensor(
                [prefix_token_ids],
                device=prompt_ids.device,
                dtype=prompt_ids.dtype,
            )
            continuation_prompt_ids = torch.cat([prompt_ids, prefix_tensor], dim=1)
            continuation_mask = torch.ones_like(continuation_prompt_ids)
            continuation_token_ids, continuation_hidden_state_steps = self._decode_steps(
                continuation_prompt_ids,
                continuation_mask,
                generation_config,
                max_new_tokens_override=remaining_tokens,
            )
            full_token_ids = prefix_token_ids + continuation_token_ids
            full_hidden_state_steps = (
                prefix_hidden_state_steps + continuation_hidden_state_steps
            )
            full_analysis = self._analysis_from_steps(
                formatted_prompt,
                full_token_ids,
                full_hidden_state_steps,
            )
            rank_score = (
                prefix_candidate.prefix_analysis.dtr
                if think_config.rank_by_prefix_dtr
                else full_analysis.dtr
            )
            completed_by_index[candidate_index] = CandidateSample(
                rank_score=rank_score,
                analysis=full_analysis,
                prefix_analysis=prefix_candidate.prefix_analysis,
                generated_tokens=len(full_token_ids),
                was_pruned=False,
            )

        final_candidates: list[CandidateSample] = []
        for index, candidate in enumerate(prefix_candidates):
            final_candidates.append(completed_by_index.get(index, candidate))
        return final_candidates

    @torch.no_grad()
    def analyze_generation(
        self,
        prompt: str,
        generated_ids: torch.Tensor,
    ) -> SequenceAnalysis:
        """Replay caller-supplied generated token ids and compute DTR."""

        formatted_prompt, prompt_ids, attention_mask = self._encode_prompt(prompt)
        token_ids, hidden_state_steps = self._decode_steps(
            prompt_ids,
            attention_mask,
            GenerationConfig(do_sample=False, max_new_tokens=int(generated_ids.numel())),
            forced_generated_ids=generated_ids.to(self.device),
        )
        return self._analysis_from_steps(
            formatted_prompt,
            token_ids,
            hidden_state_steps,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **generate_overrides: Any,
    ) -> SequenceAnalysis:
        """Generate once and return sequence-level DTR diagnostics."""

        config = generation_config or GenerationConfig()
        if generate_overrides:
            raise TypeError(
                "Pass generation settings through GenerationConfig; extra keyword "
                f"arguments are not supported: {sorted(generate_overrides)}"
            )
        if config.num_return_sequences != 1:
            raise ValueError(
                "DeepThinker.generate() decodes one sequence at a time; use think_at_n() for multiple samples."
            )

        formatted_prompt, prompt_ids, attention_mask = self._encode_prompt(prompt)
        token_ids, hidden_state_steps = self._decode_steps(
            prompt_ids,
            attention_mask,
            config,
        )
        return self._analysis_from_steps(
            formatted_prompt,
            token_ids,
            hidden_state_steps,
        )

    @torch.no_grad()
    def sample_candidates(
        self,
        prompt: str,
        think_config: Optional[ThinkAtNConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> list[CandidateSample]:
        """Sample multiple candidates and optionally prune after a prefix."""

        think_cfg = think_config or ThinkAtNConfig()
        gen_cfg = generation_config or GenerationConfig()
        formatted_prompt, prompt_ids, attention_mask = self._encode_prompt(prompt)

        if think_cfg.early_prune_by_prefix:
            return self._sample_candidates_with_early_pruning(
                formatted_prompt,
                prompt_ids,
                attention_mask,
                think_cfg,
                gen_cfg,
            )
        return self._sample_candidates_full_generation(
            formatted_prompt,
            prompt_ids,
            attention_mask,
            think_cfg,
            gen_cfg,
        )

    @torch.no_grad()
    def think_at_n(
        self,
        prompt: str,
        think_config: Optional[ThinkAtNConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> ThinkAtNResult:
        """Run Think@n repeated sampling and return the selected result."""

        think_cfg = think_config or ThinkAtNConfig()
        candidates = self.sample_candidates(
            prompt=prompt,
            think_config=think_cfg,
            generation_config=generation_config,
        )
        top_k = pick_top_k(len(candidates), think_cfg.select_top_k)
        selected = select_candidates(
            [candidate for candidate in candidates if not candidate.was_pruned],
            top_k=top_k,
        )
        if think_cfg.do_majority_vote:
            final_text, votes = majority_vote_text(
                selected,
                extractor=think_cfg.vote_text_extractor,
            )
        else:
            final_text = selected[0].analysis.text if selected else ""
            votes = {final_text: 1} if final_text else {}
        cost = estimated_think_at_n_cost(candidates)
        return ThinkAtNResult(
            final_text=final_text,
            selected_candidates=selected,
            all_candidates=candidates,
            vote_distribution=votes,
            estimated_cost_tokens=cost,
        )

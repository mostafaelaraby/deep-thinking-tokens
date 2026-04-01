from __future__ import annotations

from types import SimpleNamespace

import torch

from deep_think import DeepThinker, ThinkAtNConfig
from deep_think.config import DTRConfig, GenerationConfig


class DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt"):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens: bool = True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return " ".join(f"tok{token_id}" for token_id in token_ids).strip()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


class IdentityHead(torch.nn.Module):
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = IdentityHead()
        self.device = torch.device("cpu")
        self.forward_calls: list[int] = []

    def parameters(self):
        return iter([torch.zeros(1)])

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        self.forward_calls.append(int(input_ids.shape[1]))
        seq_len = input_ids.shape[1]
        base = torch.linspace(0.1, 0.9, steps=seq_len * 3, dtype=torch.float32).view(1, seq_len, 3)
        hidden_states = (
            base,
            base + 0.1,
            base + 0.2,
        )
        return SimpleNamespace(hidden_states=hidden_states)



def test_deep_thinker_generate_returns_analysis() -> None:
    thinker = DeepThinker(DummyModel(), DummyTokenizer(), DTRConfig(include_embedding_layer=True))
    analysis = thinker.generate(
        "hello",
        generation_config=GenerationConfig(max_new_tokens=3, do_sample=False),
    )
    assert analysis.text
    assert analysis.token_ids == [2, 2, 2]
    assert 0.0 <= analysis.dtr <= 1.0



def test_think_at_n_returns_selected_candidates() -> None:
    thinker = DeepThinker(DummyModel(), DummyTokenizer(), DTRConfig(include_embedding_layer=True))
    result = thinker.think_at_n(
        "hello",
        think_config=ThinkAtNConfig(num_samples=4, prefix_tokens=2, select_top_k=2),
        generation_config=GenerationConfig(max_new_tokens=3, do_sample=False),
    )
    assert len(result.all_candidates) == 4
    assert len(result.selected_candidates) == 2
    assert result.estimated_cost_tokens == 12


def test_think_at_n_can_vote_on_extracted_answers() -> None:
    thinker = DeepThinker(DummyModel(), DummyTokenizer(), DTRConfig(include_embedding_layer=True))
    result = thinker.think_at_n(
        "hello",
        think_config=ThinkAtNConfig(
            num_samples=4,
            prefix_tokens=2,
            select_top_k=4,
            vote_text_extractor=lambda analysis: "even" if analysis.token_ids[-1] % 2 == 0 else "odd",
        ),
        generation_config=GenerationConfig(max_new_tokens=3, do_sample=False),
    )
    assert result.final_text
    assert result.vote_distribution == {"even": 4}



def test_think_at_n_early_pruning_uses_two_stage_generation() -> None:
    model = DummyModel()
    thinker = DeepThinker(model, DummyTokenizer(), DTRConfig(include_embedding_layer=True))
    result = thinker.think_at_n(
        "hello",
        think_config=ThinkAtNConfig(
            num_samples=4,
            prefix_tokens=2,
            select_top_k=2,
            early_prune_by_prefix=True,
        ),
        generation_config=GenerationConfig(max_new_tokens=5, do_sample=False),
    )

    assert len(result.selected_candidates) == 2
    assert sum(1 for candidate in result.all_candidates if candidate.was_pruned) == 2
    assert result.estimated_cost_tokens == 14
    assert len(model.forward_calls) == 14
    assert model.forward_calls[:8] == [2, 3, 2, 3, 2, 3, 2, 3]
    assert model.forward_calls[8:] == [4, 5, 6, 4, 5, 6]



def test_analyze_generation_replays_supplied_tokens() -> None:
    thinker = DeepThinker(DummyModel(), DummyTokenizer(), DTRConfig(include_embedding_layer=True))
    analysis = thinker.analyze_generation("hello", torch.tensor([2, 0]))
    assert analysis.token_ids == [2, 0]
    assert len(analysis.token_traces) == 2

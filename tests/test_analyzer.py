from __future__ import annotations

from types import SimpleNamespace

import torch

from deep_think.analyzer import SequenceAnalyzer
from deep_think.config import DTRConfig


class DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def decode(self, token_ids, skip_special_tokens: bool = True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(str(token_id) for token_id in token_ids).strip()


class IdentityHead(torch.nn.Module):
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = IdentityHead()
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        layer0 = torch.tensor(
            [
                [
                    [0.1, 0.9],
                    [0.3, 0.7],
                    [0.8, 0.2],
                ]
            ],
            dtype=torch.float32,
        )
        layer1 = torch.tensor(
            [
                [
                    [0.2, 0.8],
                    [0.6, 0.4],
                    [0.9, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
        layer2 = torch.tensor(
            [
                [
                    [0.2, 0.8],
                    [0.9, 0.1],
                    [0.95, 0.05],
                ]
            ],
            dtype=torch.float32,
        )
        return SimpleNamespace(hidden_states=(layer0, layer1, layer2))



def test_sequence_analyzer_uses_previous_position_for_generated_token() -> None:
    model = DummyModel()
    tokenizer = DummyTokenizer()
    analyzer = SequenceAnalyzer(model, tokenizer, DTRConfig(include_embedding_layer=True))

    prompt_ids = torch.tensor([[10]])
    generated_ids = torch.tensor([[20, 21]])
    analysis = analyzer.analyze("prompt", prompt_ids, generated_ids)

    assert analysis.token_ids == [20, 21]
    assert len(analysis.token_traces) == 2
    assert all(depth >= 0 for depth in analysis.settling_depths)


def test_sequence_analyzer_excludes_tokenizer_special_tokens_from_aggregate_metrics() -> None:
    model = DummyModel()
    tokenizer = DummyTokenizer()
    analyzer = SequenceAnalyzer(model, tokenizer, DTRConfig(include_embedding_layer=True))

    prompt_ids = torch.tensor([[10]])
    analysis_with_eos = analyzer.analyze("prompt", prompt_ids, torch.tensor([[20, 0]]))
    analysis_without_eos = analyzer.analyze("prompt", prompt_ids, torch.tensor([[20]]))

    assert analysis_with_eos.token_ids == [20, 0]
    assert len(analysis_with_eos.token_traces) == 2
    assert analysis_with_eos.dtr == analysis_without_eos.dtr
    assert analysis_with_eos.mean_jsd == analysis_without_eos.mean_jsd
    assert analysis_with_eos.mean_entropy == analysis_without_eos.mean_entropy

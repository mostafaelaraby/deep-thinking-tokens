from __future__ import annotations

from deep_think import DTRConfig, GenerationConfig, ThinkAtNConfig



def test_config_defaults_are_reasonable() -> None:
    assert DTRConfig().settling_threshold == 0.5
    assert DTRConfig().apply_final_norm is True
    assert GenerationConfig().max_new_tokens > 0
    assert ThinkAtNConfig().num_samples > 0
    assert ThinkAtNConfig().early_prune_by_prefix is False

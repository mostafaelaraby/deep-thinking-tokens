from __future__ import annotations

import torch

from deep_think.metrics import classify_token, jsd_to_final, settling_depth



def test_jsd_zero_for_identical_distributions() -> None:
    p = torch.tensor([0.2, 0.3, 0.5])
    assert jsd_to_final(p, p) == 0.0



def test_settling_depth_uses_running_minimum_crossing_from_paper() -> None:
    jsds = [0.9, 0.4, 0.6, 0.2]
    assert settling_depth(jsds, settling_threshold=0.5) == 1



def test_classify_token_uses_actual_layer_indices_for_depth_cutoff() -> None:
    depth, is_deep = classify_token(
        [0.9, 0.8, 0.4],
        settling_threshold_value=0.5,
        depth_fraction=0.75,
        layer_indices=[4, 8, 12],
        total_layers=12,
    )
    assert depth == 12
    assert is_deep is True

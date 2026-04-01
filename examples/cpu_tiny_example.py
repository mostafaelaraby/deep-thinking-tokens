"""CPU-friendly example showing how DTR ranks multiple predictions."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deep_think import DeepThinker, DTRConfig, GenerationConfig, ThinkAtNConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Who discovered penicillin?"

GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=64,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
)
THINK_CONFIG = ThinkAtNConfig(
    num_samples=4,
    prefix_tokens=20,
    select_top_k=10,
    do_majority_vote=False,
    early_prune_by_prefix=False,
)


def compact_text(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:120]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(torch.device("cpu"))
    model.eval()

    thinker = DeepThinker(
        model=model,
        tokenizer=tokenizer,
        dtr_config=DTRConfig(
            settling_threshold=0.5,
            depth_fraction=0.85,
            top_k_logits=128,
            include_embedding_layer=False,
            every_k_layers=1,
        ),
    )

    print(f"Model: {MODEL_NAME}")
    print("Prompt:")
    print(PROMPT)
    print()
    print(
        "This example samples multiple predictions for the same prompt and compares their prefix DTR. "
        "That is closer to the intended Think@n usage than treating DTR as a direct hardness score."
    )
    print()

    candidates = thinker.sample_candidates(
        PROMPT,
        think_config=THINK_CONFIG,
        generation_config=GENERATION_CONFIG,
    )

    print("=== Candidates ===")
    for index, candidate in enumerate(candidates, start=1):
        print(f"Candidate {index}")
        print(f"text: {compact_text(candidate.analysis.text)}")
        print(f"prefix_dtr: {candidate.prefix_analysis.dtr:.3f}")
        print(f"full_dtr:   {candidate.analysis.dtr:.3f}")
        print(f"rank_score: {candidate.rank_score:.3f}")
        print()

    result = thinker.think_at_n(
        PROMPT,
        think_config=THINK_CONFIG,
        generation_config=GENERATION_CONFIG,
    )

    print("=== Selected ===")
    if result.selected_candidates:
        selected = result.selected_candidates[0]
        print(f"selected_text: {compact_text(selected.analysis.text)}")
        print(f"selected_prefix_dtr: {selected.prefix_analysis.dtr:.3f}")
        print(f"selected_full_dtr:   {selected.analysis.dtr:.3f}")
    else:
        print("No candidate selected.")


if __name__ == "__main__":
    main()

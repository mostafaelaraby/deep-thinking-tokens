"""Example usage with a Hugging Face causal LM."""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from deep_think import DTRConfig, DeepThinker, GenerationConfig, ThinkAtNConfig



def main() -> None:
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    thinker = DeepThinker(
        model=model,
        tokenizer=tokenizer,
        dtr_config=DTRConfig(
            settling_threshold=0.5,
            depth_fraction=0.85,
            top_k_logits=128,
            every_k_layers=1,
        ),
    )

    prompt = "Solve step by step: If 3x - 5 = 16, what is x?"
    analysis = thinker.generate(
        prompt,
        generation_config=GenerationConfig(max_new_tokens=256, do_sample=True, temperature=0.7),
    )
    print("Single sample")
    print(analysis.text)
    print(f"DTR={analysis.dtr:.3f} mean_jsd={analysis.mean_jsd:.3f}")

    think_result = thinker.think_at_n(
        prompt,
        think_config=ThinkAtNConfig(
            num_samples=8,
            prefix_tokens=20,
            select_top_k=4,
            early_prune_by_prefix=True,
        ),
        generation_config=GenerationConfig(max_new_tokens=256, do_sample=True, temperature=0.7),
    )
    print("\nThink@n with staged pruning")
    print(think_result.final_text)
    print(f"Estimated cost tokens: {think_result.estimated_cost_tokens}")
    print(f"Pruned candidates: {sum(1 for candidate in think_result.all_candidates if candidate.was_pruned)}")


if __name__ == "__main__":
    main()

# deep-thinking-tokens

[![Repository](https://img.shields.io/badge/github-mostafaelaraby%2Fdeep--thinking--tokens-black)](https://github.com/mostafaelaraby/deep-thinking-tokens)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/mostafaelaraby/deep-thinking-tokens/blob/main/LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-coming_soon-orange)](#)

`deep-thinking-tokens` is a lightweight Python library for measuring Deep-Thinking Ratio (DTR) in autoregressive language models and using it for Think@n test-time scaling.

The core idea from the paper is simple: longer chains of thought are not a reliable proxy for better reasoning, but tokens whose predicted distributions keep changing until deeper layers are stronger evidence that the model is actually spending inference-time compute on difficult decisions. This library exposes that signal in a form that fits naturally into Hugging Face workflows.

## Why use this

- Measure layer-wise reasoning effort with a reusable DTR analyzer.
- Compute DTR directly from the hidden states collected during token-by-token decoding.
- Score full generations or short prefixes using the same API.
- Run Think@n selection over multiple candidates without rewriting your inference stack.
- Optionally prune low-ranked candidates after a short prefix to reduce generation cost.

In the paper, Think@n matches or exceeds standard self-consistency while roughly halving inference cost by ranking candidates with DTR estimated from short prefixes.

## Installation

```bash
pip install git+https://github.com/mostafaelaraby/deep-thinking-tokens.git
```

For local development:

```bash
git clone https://github.com/mostafaelaraby/deep-thinking-tokens.git
cd deep-thinking-tokens
pip install -e ".[dev]"
```

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from deep_think import DTRConfig, DeepThinker, GenerationConfig, ThinkAtNConfig

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
    ),
)

analysis = thinker.generate(
    "Solve step by step: If 3x - 5 = 16, what is x?",
    generation_config=GenerationConfig(max_new_tokens=256, do_sample=True),
)
print(analysis.text)
print(analysis.dtr)

result = thinker.think_at_n(
    "Solve step by step: If 3x - 5 = 16, what is x?",
    think_config=ThinkAtNConfig(
        num_samples=8,
        prefix_tokens=20,
        select_top_k=4,
        early_prune_by_prefix=True,
    ),
    generation_config=GenerationConfig(max_new_tokens=256, do_sample=True),
)
print(result.final_text)
print(result.estimated_cost_tokens)
```

## CPU smoke test

For a quick local CPU-only sanity check with a very small model:

```bash
python examples/cpu_tiny_example.py
```

That example uses `Qwen/Qwen2.5-1.5B-Instruct`, samples multiple continuations for the same prompt, prints each candidate's DTR, and shows which output Think@n would select. This reflects the intended use of DTR as a ranking signal across predictions.

## Limitations

- **Architecture support is partial.** The logit-lens helpers look for a small set of common LM head and final-normalization module names (`lm_head`, `model.norm`, `transformer.ln_f`, etc.). Models with non-standard naming will require adapter code or manual wiring.
- **O(n²) decoding without KV cache.** Each generation step feeds the full sequence through the model because DTR requires hidden states from all layers at every step. HuggingFace models typically do not return full hidden states with cached inference, so KV cache reuse is not currently supported.
- **`apply_final_norm=True` is a deliberate deviation from the paper.** The paper's Eq. 1 defines `z_{t,l} = W_U h_{t,l}` (raw projection). This library defaults to applying the model's final LayerNorm before the unembedding projection at all layers, producing a "tuned lens" variant that tends to give more stable intermediate distributions. Set `apply_final_norm=False` for strict paper-faithful behavior.
- **`top_k_logits` is a library extension not in the paper.** The optional restriction of JSD computation to top-k vocabulary entries is provided for computational efficiency on large vocabularies but is not part of the original DTR formulation. It is disabled by default.
- **No KV cache reuse during early pruning.** The staged early-pruning path recomputes hidden states from `prompt + kept_prefix` for the second stage. This keeps the implementation simple but increases cost relative to a cache-aware approach.
- **Think@n cannot recover from universally wrong samples.** DTR ranks candidates by reasoning effort; it does not guarantee correctness if all sampled candidates arrive at the wrong answer.
- **String-based majority voting.** The default vote aggregation compares full output strings. For benchmark-faithful answer extraction, pass `ThinkAtNConfig(vote_text_extractor=...)` to normalize each completion to its final answer before voting. Tie-breaking favors the longer answer text.
- **Prompt formatting falls back silently.** If the tokenizer does not provide a compatible `apply_chat_template()`, the library falls back to raw plain-text prompts without warning, which can materially change model behavior.

## How it works

`DeepThinker.generate(...)` performs token-by-token decoding from the prompt. At each decoding step it keeps the model's hidden states, projects the selected layers through the LM head, compares each intermediate layer to the final layer with JSD, and then computes the settling depth for the generated token.

Sequence-level DTR is the fraction of generated tokens whose settling depth falls in the late-depth regime controlled by `depth_fraction`.

## Think@n modes

`DeepThinker.think_at_n(...)` supports two modes:

- Full generation ranking (default)
  - Generate each sample to completion, rank candidates by full-sequence DTR.
- Staged early pruning
  - Set `early_prune_by_prefix=True` to generate only the first `prefix_tokens`, rank candidates by prefix DTR, discard low-ranked candidates, and continue generation only for the selected subset.

By default, candidates are ranked by **full-sequence DTR**. Set `rank_by_prefix_dtr=True` to rank by DTR computed on the first `prefix_tokens` instead.

Example:

```python
# Full generation, ranked by full DTR (default)
result = thinker.think_at_n(
    prompt,
    think_config=ThinkAtNConfig(
        num_samples=8,
        select_top_k=4,
    ),
    generation_config=GenerationConfig(max_new_tokens=256, do_sample=True),
)

# Staged early pruning, ranked by prefix DTR
result = thinker.think_at_n(
    prompt,
    think_config=ThinkAtNConfig(
        num_samples=8,
        prefix_tokens=20,
        select_top_k=4,
        rank_by_prefix_dtr=True,
        early_prune_by_prefix=True,
    ),
    generation_config=GenerationConfig(max_new_tokens=256, do_sample=True),
)
```

Notes:

- `prefix_tokens=20` means the early-pruning decision is made after the first 20 generated tokens.
- When staged pruning is enabled, pruned candidates stop at that prefix instead of being fully generated. The initial pruning decision always uses prefix DTR (full DTR is not yet available); `rank_by_prefix_dtr` controls the `rank_score` on the final candidates used for selection.
- The continuation path is intentionally simple and recomputes hidden states from `prompt + kept_prefix` for the second stage instead of using KV-cache reuse.

## API overview

`DeepThinker` is the main integration point.

- `generate(...)`
  - Run one generation and return a `SequenceAnalysis`.
- `analyze_generation(...)`
  - Replay caller-supplied generated token ids and compute DTR from the corresponding decoding steps.
- `sample_candidates(...)`
  - Sample multiple candidates and score them with prefix DTR.
- `think_at_n(...)`
  - Rank candidates by DTR, keep the best subset, and aggregate them.

Key result objects:

- `SequenceAnalysis`
  - Generated text, token ids, DTR, mean JSD, mean entropy, per-token settling depths, and token traces.
- `ThinkAtNResult`
  - Final selected text, ranked candidates, vote counts, and estimated token cost.
- `CandidateSample`
  - Per-candidate full analysis, prefix analysis, rank score, and whether that candidate was pruned early.

## Design notes

The package is split into three layers:

- `deep_think.logit_lens`
  - Model-facing logic for hidden-state projection and architecture-specific normalization.
- `deep_think.metrics`
  - Pure DTR math, including JSD, settling depth, and sequence aggregation.
- `deep_think.generation` and `deep_think.aggregation`
  - Online decoding, DTR integration, and Think@n orchestration.

This separation keeps the metric testable and makes it easier to integrate into existing inference code.

## Examples

- `examples/example.py`
  - Larger Hugging Face example.
- `examples/cpu_tiny_example.py`
  - Small CPU-friendly example for local validation.

## Repository

- GitHub: https://github.com/mostafaelaraby/deep-thinking-tokens

## Citation

If you use this repository, cite both the software and the original paper. The machine-readable software citation is provided in `CITATION.cff`.

```bibtex
@article{chen2026thinkdeep,
  title={Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens},
  author={Chen, Wei-Lin and Peng, Liqian and Tan, Tian and Zhao, Chao and Chen, Blake JianHang and Lin, Ziqian and Go, Alec and Meng, Yu},
  journal={arXiv preprint arXiv:2602.13517},
  year={2026}
}
```

Paper links:

- https://arxiv.org/abs/2602.13517
- https://arxiv.org/html/2602.13517v1

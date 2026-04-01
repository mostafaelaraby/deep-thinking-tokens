"""Small helpers shared across generation and analysis."""

from __future__ import annotations

from typing import Optional

from deep_think.interfaces import TokenizerLike


def safe_pad_token_id(tokenizer: TokenizerLike) -> int:
    """Return a usable pad token id for generation."""

    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def build_prompt(tokenizer: TokenizerLike, prompt: str) -> str:
    """Build an instruct-style prompt when the tokenizer supports it."""

    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def decode_text(tokenizer: TokenizerLike, token_ids: object) -> str:
    """Decode text while skipping special tokens."""

    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def pick_top_k(total: int, explicit_top_k: Optional[int]) -> int:
    """Resolve top-k sample selection for Think@n."""

    if explicit_top_k is None:
        return max(1, total // 2)
    return max(1, min(total, explicit_top_k))

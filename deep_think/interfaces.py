"""Protocol definitions used to keep integration points explicit."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class TokenizerLike(Protocol):
    """Subset of tokenizer behavior used by this package."""

    eos_token_id: Optional[int]
    pad_token_id: Optional[int]

    def __call__(self, text: str, return_tensors: str = "pt") -> Any:
        ...

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        ...

    def apply_chat_template(
        self,
        messages: Any,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        ...


@runtime_checkable
class CausalLMOutputLike(Protocol):
    """Subset of causal LM outputs used by this package."""

    hidden_states: Any


@runtime_checkable
class CausalLMModelLike(Protocol):
    """Subset of model behavior used by this package."""

    device: torch.device
    lm_head: torch.nn.Module

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> CausalLMOutputLike:
        ...

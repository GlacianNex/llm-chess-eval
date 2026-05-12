"""Single entry point for constructing an adapter from a model ID.

Routes by `provider_for_model(model_id)` so callers don't have to know which
provider owns which name.
"""
from __future__ import annotations

from ..config import provider_for_model
from .base import ModelAdapter


def build_adapter(model: str, augment_legal_moves: bool = False, max_tokens: int = 2048) -> ModelAdapter:
    """Construct the right adapter for `model`, lazily importing the SDK we need."""
    provider = provider_for_model(model)

    if provider == "anthropic":
        from .claude import ClaudeAdapter
        return ClaudeAdapter(
            model=model, max_tokens=max_tokens, augment_legal_moves=augment_legal_moves
        )
    if provider == "openai":
        from .openai import OpenAIAdapter
        return OpenAIAdapter(
            model=model, max_tokens=max_tokens, augment_legal_moves=augment_legal_moves
        )
    if provider == "google":
        from .gemini import GeminiAdapter
        return GeminiAdapter(
            model=model, max_tokens=max_tokens, augment_legal_moves=augment_legal_moves
        )
    if provider == "together":
        from .openai_compat import together_llama
        return together_llama(
            model=model, max_tokens=max_tokens, augment_legal_moves=augment_legal_moves
        )
    if provider == "groq":
        from .openai_compat import groq_llama
        return groq_llama(
            model=model, max_tokens=max_tokens, augment_legal_moves=augment_legal_moves
        )
    if provider == "deepseek":
        from .openai_compat import deepseek
        return deepseek(
            model=model, max_tokens=max_tokens, augment_legal_moves=augment_legal_moves
        )
    if provider == "ollama":
        from .openai_compat import ollama_local
        return ollama_local(
            model=model, max_tokens=max_tokens, augment_legal_moves=augment_legal_moves
        )

    raise ValueError(f"No adapter wired for provider '{provider}'")

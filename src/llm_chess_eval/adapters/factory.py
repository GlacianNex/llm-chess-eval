"""Single entry point for constructing an adapter from a model ID.

Routes by `provider_for_model(model_id)` so callers don't have to know which
provider owns which name.
"""
from __future__ import annotations

from ..config import provider_for_model
from .base import ModelAdapter


def build_adapter(model: str, augment_legal_moves: bool = False, max_tokens: int = 65536) -> ModelAdapter:
    """Construct the right adapter for `model`, lazily importing the SDK we need.

    Default max_tokens=65536 is the generous ceiling required by reasoning
    models on OpenAI and Gemini, whose response-token budget INCLUDES the
    internal reasoning tokens. Anthropic counts only output tokens (thinking
    is on a separate budget), and DeepSeek-reasoner returns reasoning in a
    separate `reasoning_content` field, so a high ceiling is harmless for
    them.

    BUG NOTE: An earlier version of this default was 2048, which silently
    capped OpenAI's reasoning models far below their actual reasoning needs.
    The individual adapter constructors had 65536 as their own default,
    but this factory-level 2048 override won. The result: GPT-5 and
    GPT-5-mini ran the entire v2 matrix at max_completion_tokens=2048
    (reasoning + output combined), which exhausted before any tool call
    could be emitted on ~98% of plies. See HANDOFF.md for the corrected
    OpenAI numbers after re-running with this fix.
    """
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

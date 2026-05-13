"""Single entry point for constructing an adapter from a model ID.

Routes by `provider_for_model(model_id)` so callers don't have to know which
provider owns which name.
"""
from __future__ import annotations

from ..config import provider_for_model
from .base import ModelAdapter


def build_adapter(model: str, augment_legal_moves: bool = False, max_tokens: int = 65536) -> ModelAdapter:
    """Construct the right adapter for `model`, lazily importing the SDK we need.

    The `-pytool` suffix is a routing flag: if present, route to the
    python-tool-augmented adapter for the same provider, and strip the
    suffix before passing the model id to the SDK. Currently wired for
    Anthropic only; other providers fall back to the standard adapter
    with a NotImplementedError if -pytool is requested for them.

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
    could be emitted on ~98% of plies. See RESULTS.md for the corrected
    OpenAI numbers after re-running with this fix.
    """
    pytool = model.endswith("-pytool")
    base_model = model[:-len("-pytool")] if pytool else model
    provider = provider_for_model(model)

    # Per-provider max_tokens ceiling. Different providers count tokens
    # differently:
    #   - Anthropic: max_tokens = OUTPUT tokens only. Thinking is a
    #     separate budget. A chess tool call needs <2k tokens, so the
    #     full 65536 ceiling is wasteful AND trips the SDK's "operation
    #     longer than 10 minutes → must use streaming" guard (which we
    #     don't currently support). Clamp to 8192 — plenty of headroom
    #     for the tool call output.
    #   - OpenAI / Gemini / DeepSeek-reasoner: max_tokens INCLUDES the
    #     internal reasoning budget on reasoning models. 65536 is the
    #     ceiling required to fit reasoning chains.
    anthropic_max_tokens = min(max_tokens, 8192)

    if provider == "anthropic":
        if pytool:
            from .claude_python import ClaudePythonAdapter
            return ClaudePythonAdapter(
                model=base_model, max_tokens=max(anthropic_max_tokens, 4096), augment_legal_moves=augment_legal_moves
            )
        from .claude import ClaudeAdapter
        return ClaudeAdapter(
            model=base_model, max_tokens=anthropic_max_tokens, augment_legal_moves=augment_legal_moves
        )
    if pytool:
        raise NotImplementedError(
            f"-pytool variant not yet wired for provider '{provider}'. "
            f"Currently supported: anthropic."
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

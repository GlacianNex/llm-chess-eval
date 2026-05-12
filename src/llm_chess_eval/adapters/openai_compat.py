"""OpenAI-compatible adapter — works for any endpoint that speaks OpenAI's API shape.

Use cases:
  - Meta Llama via Together AI (base_url=https://api.together.xyz/v1, key=TOGETHER_API_KEY)
  - Meta Llama via Groq (base_url=https://api.groq.com/openai/v1, key=GROQ_API_KEY)
  - Local Ollama (base_url=http://localhost:11434/v1, no key needed — pass "ollama")
  - Self-hosted vLLM / TGI servers
  - DeepSeek, Mistral, etc. (most providers ship OpenAI-shaped endpoints)

Tool-calling support varies. Some reasoning models (e.g. deepseek-reasoner) reject
the OpenAI-style forced specific-tool `tool_choice` and only accept `"required"`
(any tool). Some models also append trailing text after the JSON arguments, so we
parse the first complete JSON object rather than requiring the whole string to be JSON.
"""
from __future__ import annotations

import json
import os

from ._shared import (
    SUBMIT_MOVE_DESCRIPTION,
    SUBMIT_MOVE_PARAMETERS,
    SYSTEM_PROMPT,
    CallOutcome,
    Timer,
    build_user_message,
    parse_tool_input,
)

SUBMIT_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_move",
        "description": SUBMIT_MOVE_DESCRIPTION,
        "parameters": SUBMIT_MOVE_PARAMETERS,
    },
}


class OpenAICompatibleAdapter:
    """Generic adapter for any OpenAI-API-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env_var: str,
        max_tokens: int = 16000,  # generous to fit reasoning + tool output on reasoning-tier models
        augment_legal_moves: bool = False,
        provider_label: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is not installed. Run: pip install openai>=1.0"
            ) from e

        key = os.environ.get(api_key_env_var)
        if not key:
            raise RuntimeError(
                f"{api_key_env_var} is not set. Export it before running this eval."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.augment_legal_moves = augment_legal_moves
        self.provider_label = provider_label or base_url
        self._client = OpenAI(api_key=key, base_url=base_url)

    def propose_move(
        self,
        fen: str,
        prior_failed: list[str] | None = None,
        augment_legal_moves: bool | None = None,
    ) -> CallOutcome:
        use_aug = self.augment_legal_moves if augment_legal_moves is None else augment_legal_moves
        user_text = build_user_message(fen, prior_failed=prior_failed, augment_legal_moves=use_aug)

        raw_input = None
        error = None
        in_tok = out_tok = 0

        def _do_call(tool_choice):
            return self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                tools=[SUBMIT_MOVE_TOOL],
                tool_choice=tool_choice,
            )

        with Timer() as t:
            try:
                # Try increasingly permissive tool_choice forms until one is accepted.
                # Most providers accept "required" (force any tool). Some accept only
                # the OpenAI-style specific-tool object. Some reasoning models
                # (deepseek-reasoner) accept only "auto" (model decides).
                resp = None
                last_err: Exception | None = None
                for tc in (
                    "required",
                    {"type": "function", "function": {"name": "submit_move"}},
                    "auto",
                ):
                    try:
                        resp = _do_call(tc)
                        last_err = None
                        break
                    except Exception as e:  # noqa: BLE001
                        msg = str(e).lower()
                        last_err = e
                        # If the error isn't about tool_choice, stop fallback chain.
                        if "tool_choice" not in msg and "tool choice" not in msg:
                            raise
                if resp is None and last_err is not None:
                    raise last_err

                usage = resp.usage
                if usage is not None:
                    in_tok = getattr(usage, "prompt_tokens", 0) or 0
                    out_tok = getattr(usage, "completion_tokens", 0) or 0

                choice = resp.choices[0] if resp.choices else None
                tool_calls = getattr(choice.message, "tool_calls", None) if choice else None
                if not tool_calls:
                    fr = getattr(choice, "finish_reason", None) if choice else None
                    content = getattr(choice.message, "content", None) if choice else None
                    snippet = (content or "")[:120]
                    error = f"Model did not call submit_move (finish_reason={fr!r}, content_preview={snippet!r})"
                else:
                    args_str = tool_calls[0].function.arguments
                    # Lenient parse: extract first complete JSON object, ignore trailing.
                    # Some models append text after the JSON arguments.
                    try:
                        raw_input = json.loads(args_str)
                    except json.JSONDecodeError:
                        try:
                            decoder = json.JSONDecoder()
                            raw_input, _ = decoder.raw_decode(args_str.lstrip())
                        except json.JSONDecodeError as e:
                            error = f"Tool arguments are not valid JSON: {e}"
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"

        parsed = None
        if raw_input is not None and error is None:
            parsed, parse_err = parse_tool_input(raw_input)
            if parse_err:
                error = parse_err

        return CallOutcome(
            response=parsed,
            raw_tool_input=raw_input,
            latency_ms=t.ms,
            error=error,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )


# ----- Pre-configured factory for common providers -----


def together_llama(model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", **kwargs) -> OpenAICompatibleAdapter:
    """Llama via Together AI. Requires TOGETHER_API_KEY."""
    return OpenAICompatibleAdapter(
        model=model,
        base_url="https://api.together.xyz/v1",
        api_key_env_var="TOGETHER_API_KEY",
        provider_label="together",
        **kwargs,
    )


def groq_llama(model: str = "llama-3.3-70b-versatile", **kwargs) -> OpenAICompatibleAdapter:
    """Llama via Groq. Requires GROQ_API_KEY."""
    return OpenAICompatibleAdapter(
        model=model,
        base_url="https://api.groq.com/openai/v1",
        api_key_env_var="GROQ_API_KEY",
        provider_label="groq",
        **kwargs,
    )


def deepseek(model: str = "deepseek-chat", **kwargs) -> OpenAICompatibleAdapter:
    """DeepSeek via their hosted API (OpenAI-compatible). Requires DEEPSEEK_API_KEY.
    Models: 'deepseek-chat' (V3/budget), 'deepseek-reasoner' (R1/frontier)."""
    return OpenAICompatibleAdapter(
        model=model,
        base_url="https://api.deepseek.com/v1",
        api_key_env_var="DEEPSEEK_API_KEY",
        provider_label="deepseek",
        **kwargs,
    )


def ollama_local(model: str = "llama3.1:70b", **kwargs) -> OpenAICompatibleAdapter:
    """Llama (or any model) via local Ollama. No API key needed but the env var
    must exist — Ollama accepts any non-empty string."""
    os.environ.setdefault("OLLAMA_API_KEY", "ollama")
    return OpenAICompatibleAdapter(
        model=model,
        base_url="http://localhost:11434/v1",
        api_key_env_var="OLLAMA_API_KEY",
        provider_label="ollama-local",
        **kwargs,
    )

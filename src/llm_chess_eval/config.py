"""Runtime configuration: binary paths, model IDs, defaults."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

V1_MODELS = ["claude-opus-4-7", "claude-sonnet-4-6"]

# Standardized model matrix for cross-provider ranking.
# Each provider contributes one "frontier" (strongest) and one "budget" (cheapest reasonable) tier.
BENCHMARK_MATRIX: dict[str, dict[str, str]] = {
    "anthropic": {"frontier": "claude-opus-4-7",          "budget": "claude-haiku-4-5-20251001"},
    "openai":    {"frontier": "gpt-5",                    "budget": "gpt-5-mini"},
    # Google's frontier was originally gemini-3.1-pro-preview, but Google
    # enforces a 250 req/day cap on preview-track models regardless of paid
    # tier — too tight for CR+PS with retries. Use gemini-2.5-pro (GA, no
    # daily cap) as the published frontier cell.
    "google":    {"frontier": "gemini-2.5-pro",           "budget": "gemini-3.1-flash-lite"},
    "deepseek":  {"frontier": "deepseek-reasoner",        "budget": "deepseek-chat"},
}


def benchmark_models(tier: str | None = None) -> list[str]:
    """Return all benchmark-matrix models, optionally filtered to a single tier."""
    out = []
    for provider, tiers in BENCHMARK_MATRIX.items():
        if tier is None:
            out.extend(tiers.values())
        else:
            if tier in tiers:
                out.append(tiers[tier])
    return out


# Multi-provider model registry. Keyed by user-facing model ID; value is the provider key.
# Adapters are instantiated by `build_adapter(model_id)` in adapter_factory.py.
KNOWN_MODELS: dict[str, str] = {
    # Anthropic
    "claude-opus-4-7":             "anthropic",
    "claude-sonnet-4-6":           "anthropic",
    "claude-haiku-4-5-20251001":   "anthropic",
    # OpenAI
    "gpt-5":        "openai",
    "gpt-5-mini":   "openai",
    "gpt-4o":       "openai",
    "gpt-4o-mini":  "openai",
    # Google Gemini
    "gemini-3.1-pro-preview":      "google",
    "gemini-3.1-flash-lite":       "google",
    "gemini-3.1-flash-lite-preview": "google",
    "gemini-3-pro-preview":        "google",
    "gemini-pro-latest":           "google",
    "gemini-2.5-pro":              "google",
    "gemini-2.5-flash":            "google",
    # Meta / Llama via Together AI (Together is the default Llama provider)
    "meta-llama/Llama-4-Scout-17B-16E-Instruct":     "together",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo":       "together",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo":  "together",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo":   "together",
    # Llama via Groq
    "llama-3.3-70b-versatile":     "groq",
    # DeepSeek
    "deepseek-chat":               "deepseek",
    "deepseek-reasoner":           "deepseek",
}


def provider_for_model(model_id: str) -> str:
    """Resolve a model ID to its provider. Falls back to prefix matching.

    The `-pytool` suffix is a routing flag (not part of the underlying model
    ID) — the factory uses it to pick a python-tool-augmented adapter and
    strips it before passing the name to the provider. Treat it transparently
    here.
    """
    base = model_id[:-len("-pytool")] if model_id.endswith("-pytool") else model_id
    if base in KNOWN_MODELS:
        return KNOWN_MODELS[base]
    if base.startswith("claude-"):
        return "anthropic"
    if base.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    if base.startswith("gemini-"):
        return "google"
    if base.startswith("meta-llama/") or "llama" in base.lower():
        return "together"
    if base.startswith("deepseek"):
        return "deepseek"
    raise ValueError(
        f"Unknown model '{model_id}'. Add it to KNOWN_MODELS in config.py or use "
        f"a recognizable name prefix (claude-/gpt-/gemini-/meta-llama/...)."
    )

_STOCKFISH_FALLBACK = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Microsoft" / "WinGet" / "Packages"
    / "Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
)


def stockfish_path() -> Path:
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"STOCKFISH_PATH points to nonexistent file: {p}")
    on_path = shutil.which("stockfish")
    if on_path:
        return Path(on_path)
    if _STOCKFISH_FALLBACK.is_file():
        return _STOCKFISH_FALLBACK
    raise FileNotFoundError(
        "Stockfish not found. Set STOCKFISH_PATH or add stockfish to PATH."
    )


def anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running an eval."
        )
    return key

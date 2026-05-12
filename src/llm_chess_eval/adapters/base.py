"""Adapter protocol every provider implementation must satisfy."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ._shared import CallOutcome


@runtime_checkable
class ModelAdapter(Protocol):
    """All adapters expose this surface so the eval harness doesn't care which provider it's talking to."""
    model: str
    augment_legal_moves: bool

    def propose_move(
        self,
        fen: str,
        prior_failed: list[str] | None = None,
        augment_legal_moves: bool | None = None,
    ) -> CallOutcome: ...

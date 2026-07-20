# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Python-side generation defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Serializable arguments accepted by ``UniRTLLM.generate``.

    Sampler values of zero defer to the backend's default behavior. A zero
    ``n_past`` requests a stateless call by clearing prior KV state. Context
    sliding remains backend-specific and is rejected by bundled backends that
    cannot preserve it faithfully.
    """

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 0
    stop: list[str] = field(default_factory=list)
    grammar: str | None = None
    json_mode: bool = False
    images: list[str] = field(default_factory=list)
    audios: list[str] = field(default_factory=list)
    stream: bool = False
    n_past: int = 0
    sliding_window: bool = False
    sliding_window_n_keep: int = 0


__all__ = ['GenerationConfig']

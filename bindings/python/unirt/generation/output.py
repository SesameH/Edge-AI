# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Generation result and profiling value objects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .._ffi._types import unirt_ProfileData

_THINK_BLOCK = re.compile(r'<think>(.*?)</think>', flags=re.DOTALL)


def _format_us(value: int) -> str:
    if value <= 0:
        return '0 µs'
    if value < 1_000:
        return f'{value} µs'
    if value < 1_000_000:
        return f'{value / 1_000:.1f} ms'
    return f'{value / 1_000_000:.2f} s'


@dataclass(repr=False)
class GenerationProfile:
    """Timing, throughput, stop cause, and model provenance for one call."""

    ttft: int = 0
    prompt_time: int = 0
    decode_time: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_speed: float = 0.0
    decode_speed: float = 0.0
    stop_reason: str | None = None
    backend: str | None = None
    device: str | None = None
    quant: str | None = None
    model_path: str | None = None

    @classmethod
    def from_c(cls, native: unirt_ProfileData) -> 'GenerationProfile':
        reason = native.stop_reason.decode('utf-8', errors='replace') if native.stop_reason else None
        return cls(
            ttft=int(native.ttft),
            prompt_time=int(native.prompt_time),
            decode_time=int(native.decode_time),
            prompt_tokens=int(native.prompt_tokens),
            generated_tokens=int(native.generated_tokens),
            prefill_speed=float(native.prefill_speed),
            decode_speed=float(native.decoding_speed),
            stop_reason=reason,
        )

    def __repr__(self) -> str:
        values = (
            f'ttft={_format_us(self.ttft)}',
            f'prompt_time={_format_us(self.prompt_time)}',
            f'decode_time={_format_us(self.decode_time)}',
            f'prompt_tokens={self.prompt_tokens} tok',
            f'generated_tokens={self.generated_tokens} tok',
            f'prefill_speed={self.prefill_speed:.1f} tok/s',
            f'decode_speed={self.decode_speed:.1f} tok/s',
            f'stop_reason={self.stop_reason}',
            f'backend={self.backend}',
            f'device={self.device}',
            f'quant={self.quant}',
            f'model_path={self.model_path}',
        )
        return f'GenerationProfile({", ".join(values)})'


@dataclass
class GenerateOutput:
    """Visible response text, optional reasoning block, and profile data."""

    text: str = ''
    thinking: str | None = None
    profile: GenerationProfile = field(default_factory=GenerationProfile)

    @classmethod
    def from_raw(cls, full_text: str, profile: GenerationProfile) -> 'GenerateOutput':
        match = _THINK_BLOCK.search(full_text)
        if match is None:
            return cls(text=full_text, profile=profile)
        visible = _THINK_BLOCK.sub('', full_text).strip()
        return cls(text=visible, thinking=match.group(1).strip(), profile=profile)


__all__ = ['GenerateOutput', 'GenerationProfile']

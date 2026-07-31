# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Curated starting points, sized against the machine that will run them.

Picking a first model is two questions the user should not have to research:
which checkpoints actually load on a given backend, and which of those fit in
RAM. The second is arithmetic -- ``download_gib`` plus room for the KV cache and
the rest of the system -- so it lives in ``fits_in``. The first is not: the MLX
plugin accepts a narrow slice of Hugging Face (``model_type=llama``, tied
embeddings, and a ``[Digits, ByteLevel]`` BPE tokenizer), and a checkpoint that
misses those is rejected at load rather than silently degraded. Every entry here
was checked against the running backend, so the list stays short on purpose:
a wrong recommendation costs a multi-gigabyte download to discover.

Sizes are the real file sizes on the Hub as of 2026-07, not estimates.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass

__all__ = ['Entry', 'CATALOG', 'aliases', 'recommend', 'total_memory_gib']

# Headroom over the weights: KV cache at a few thousand tokens, the runtime
# itself, and whatever else the machine is doing. Undersizing this is the
# failure mode that matters -- a model that swaps is worse than one that is a
# size smaller, as anything paging to disk stops being bandwidth-bound at all.
# Encoders keep no KV cache and run one short sequence at a time, so they get a
# far smaller allowance than a generative model of the same weight size would.
_HEADROOM_GIB = {'embed': 0.5}
_DEFAULT_HEADROOM_GIB = 2.0


@dataclass(frozen=True)
class Entry:
    """One recommended checkpoint."""

    alias: str
    repo: str
    backend: str
    task: str
    download_gib: float
    note: str
    quant: str | None = None

    @property
    def ref(self) -> str:
        """The argument to pass to ``pull``/``chat``."""
        return f'{self.repo}:{self.quant}' if self.quant else self.repo

    @property
    def min_ram_gib(self) -> float:
        headroom = _HEADROOM_GIB.get(self.task, _DEFAULT_HEADROOM_GIB)
        return round(self.download_gib + headroom, 1)

    def fits_in(self, ram_gib: float) -> bool:
        return ram_gib >= self.min_ram_gib


# Ordered smallest-first within each task so `recommend` degrades gracefully on
# small machines: truncating the list still leaves something runnable.
CATALOG: tuple[Entry, ...] = (
    # --- chat, llama_cpp (GGUF): the broad path, any architecture llama.cpp knows.
    Entry('gemma3-270m', 'ggml-org/gemma-3-270m-it-GGUF', 'llama_cpp', 'chat', 0.27,
          'Smallest useful chat model; good for smoke tests and CI.', 'Q8_0'),
    Entry('llama3.2-1b', 'unsloth/Llama-3.2-1B-Instruct-GGUF', 'llama_cpp', 'chat', 0.75,
          'Solid general chat at 1B; 128k context.', 'Q4_K_M'),
    Entry('gemma3-1b', 'ggml-org/gemma-3-1b-it-GGUF', 'llama_cpp', 'chat', 0.75,
          'Strong instruction following for its size.', 'Q4_K_M'),
    Entry('smollm2-1.7b', 'unsloth/SmolLM2-1.7B-Instruct-GGUF', 'llama_cpp', 'chat', 0.98,
          'Fully open training data; the MLX backend runs it too.', 'Q4_K_M'),
    Entry('qwen3-1.7b', 'ggml-org/Qwen3-1.7B-GGUF', 'llama_cpp', 'chat', 1.19,
          'Reasoning-capable, multilingual; good default under 2 GiB.', 'Q4_K_M'),
    Entry('llama3.2-3b', 'unsloth/Llama-3.2-3B-Instruct-GGUF', 'llama_cpp', 'chat', 1.88,
          'Noticeably better than 1B at summarising and following format.', 'Q4_K_M'),
    Entry('qwen3-4b', 'ggml-org/Qwen3-4B-GGUF', 'llama_cpp', 'chat', 2.33,
          'Best quality that still fits comfortably in 8 GiB.', 'Q4_K_M'),
    Entry('qwen3-8b', 'ggml-org/Qwen3-8B-GGUF', 'llama_cpp', 'chat', 4.68,
          'Wants 16 GiB and patience; the quality step over 4B is real.', 'Q4_K_M'),

    # --- code
    Entry('coder-1.5b', 'Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF', 'llama_cpp', 'code', 1.04,
          'Code completion and short edits.', 'Q4_K_M'),
    Entry('coder-7b', 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF', 'llama_cpp', 'code', 4.36,
          'Competitive with much larger general models on code.', 'Q4_K_M'),

    # --- vision (VLM via mtmd; the mmproj projector downloads alongside).
    Entry('smolvlm-500m', 'ggml-org/SmolVLM-500M-Instruct-GGUF', 'llama_cpp', 'vlm', 0.51,
          'Smallest vision-language model worth running.', 'Q8_0'),
    Entry('qwen2.5-vl-3b', 'ggml-org/Qwen2.5-VL-3B-Instruct-GGUF', 'llama_cpp', 'vlm', 2.59,
          'Reads documents, charts and screenshots well.', 'Q4_K_M'),
    Entry('gemma3-4b', 'ggml-org/gemma-3-4b-it-GGUF', 'llama_cpp', 'vlm', 2.32,
          'Chat and vision in one checkpoint.', 'Q4_K_M'),

    # --- embeddings (ONNX Runtime)
    # No quant token: ONNX variants are chosen at encode time via
    # `unirt embed --precision`, not by a repo:precision pull target.
    Entry('minilm', 'sentence-transformers/all-MiniLM-L6-v2', 'onnxruntime', 'embed', 0.09,
          'Sentence embeddings; add --precision qint8_arm64 for the 23 MiB build.'),

    # --- chat, mlx (Apple Silicon, HF safetensors).
    # This list is short because the plugin's gates are narrow, not because
    # nothing else is good: Qwen, Gemma, TinyLlama and Llama-3.x all fail one of
    # model_type / tied embeddings / tokenizer shape. Use llama_cpp for those.
    Entry('smollm2-135m-mlx', 'HuggingFaceTB/SmolLM2-135M-Instruct', 'mlx', 'chat', 0.26,
          'Unquantized; fastest way to check the MLX path works.'),
    Entry('smollm2-360m-mlx', 'HuggingFaceTB/SmolLM2-360M-Instruct', 'mlx', 'chat', 0.69,
          'Unquantized bf16 weights, cast to fp16 at load.'),
    Entry('smollm2-1.7b-mlx', 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'mlx', 'chat', 3.42,
          'The largest checkpoint the MLX backend currently accepts.'),
)


def aliases() -> dict[str, str]:
    """Alias table for ``model_manager``, mapping short name to pull target."""
    return {entry.alias: entry.ref for entry in CATALOG}


def total_memory_gib() -> float | None:
    """Physical RAM in GiB, or None when the platform will not say."""
    try:
        if sys.platform == 'win32':
            class _Status(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.ullTotalPhys / 2**30
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
    except (AttributeError, OSError, ValueError):
        return None


def recommend(
    *,
    ram_gib: float | None = None,
    backend: str | None = None,
    task: str | None = None,
    include_oversized: bool = False,
) -> list[Entry]:
    """Catalog entries matching the filters, largest (best) first per task.

    ``ram_gib=None`` means "do not filter by size" -- the caller could not
    determine memory, so listing everything beats guessing wrong.
    """
    selected = [
        entry for entry in CATALOG
        if (backend is None or entry.backend == backend)
        and (task is None or entry.task == task)
        and (include_oversized or ram_gib is None or entry.fits_in(ram_gib))
    ]
    # Group by task in catalog order, then biggest-first inside each group so
    # the best thing that fits reads off the top of its section.
    task_order: dict[str, int] = {}
    for index, entry in enumerate(CATALOG):
        task_order.setdefault(entry.task, index)
    return sorted(selected, key=lambda e: (task_order[e.task], -e.download_gib))

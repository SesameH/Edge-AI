# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Normalization and tqdm rendering for model-download progress."""

from __future__ import annotations

from typing import Callable

from tqdm.auto import tqdm

from .model_manager import DownloadProgress, ProgressCallback


class _TqdmBar:
    def __init__(self) -> None:
        self._bars: dict[str, object] = {}

    def __call__(self, files: list[DownloadProgress]) -> bool:
        for item in files:
            bar = self._bars.get(item.file_name)
            if bar is None:
                bar = tqdm(
                    total=item.total_bytes if item.total_bytes > 0 else None,
                    initial=item.downloaded_bytes,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=item.file_name,
                    leave=True,
                )
                self._bars[item.file_name] = bar
            elif item.total_bytes > 0 and bar.total != item.total_bytes:
                bar.total = item.total_bytes
                bar.refresh()
            bar.update(item.downloaded_bytes - bar.n)
        return True

    def finish(self) -> None:
        bars, self._bars = self._bars, {}
        for bar in bars.values():
            bar.close()


def default_progress_printer() -> ProgressCallback:
    return _TqdmBar()


def resolve(progress: ProgressCallback | bool | None) -> ProgressCallback | None:
    if progress is False:
        return None
    if progress is None:
        return default_progress_printer()
    if callable(progress):
        return progress
    raise TypeError(
        f'progress must be callable, False, or None; got {type(progress).__name__}'
    )


def finish(printer: Callable[..., bool] | None) -> None:
    closer = getattr(printer, 'finish', None) if printer is not None else None
    if callable(closer):
        closer()


__all__ = ['default_progress_printer', 'resolve', 'finish']

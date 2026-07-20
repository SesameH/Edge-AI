# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Thread-backed iterator used by streaming native generation."""

from __future__ import annotations

import codecs
import queue
import threading
from ctypes import c_void_p
from typing import Callable, Iterator

from .._ffi._types import unirt_token_callback
from .output import GenerateOutput

_END = object()


class TextIteratorStreamer:
    """Yield decoded UTF-8 chunks while generation runs on a worker thread."""

    def __init__(self) -> None:
        self._items: queue.Queue[object] = queue.Queue()
        self._output: GenerateOutput | None = None
        self._error: BaseException | None = None
        self._worker: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._cb_ref: unirt_token_callback | None = None
        self._decoder_finalizer: Callable[[], str] | None = None

    @property
    def output(self) -> GenerateOutput | None:
        return self._output

    def cancel(self) -> None:
        """Ask the native callback to stop at its next token boundary."""
        self._cancel_event.set()

    def _make_callback(self) -> unirt_token_callback:
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        @unirt_token_callback
        def callback(token_bytes: bytes | None, _user_data: c_void_p) -> bool:
            if token_bytes is not None:
                chunk = decoder.decode(token_bytes, final=False)
                if chunk:
                    self._items.put(chunk)
            return not self._cancel_event.is_set()

        def finish_decoder() -> str:
            pending, _state = decoder.getstate()
            if pending:
                decoder.reset()
                return ''
            return decoder.decode(b'', final=True)

        self._cb_ref = callback
        self._decoder_finalizer = finish_decoder
        return callback

    def _execute(self, generate: Callable[[], GenerateOutput]) -> None:
        try:
            self._output = generate()
        except BaseException as exc:  # propagate worker errors from iteration
            self._error = exc
        finally:
            if self._decoder_finalizer is not None:
                tail = self._decoder_finalizer()
                if tail:
                    self._items.put(tail)
            self._items.put(_END)

    def start(self, generate: Callable[[], GenerateOutput]) -> None:
        self._worker = threading.Thread(
            target=self._execute,
            args=(generate,),
            name='unirt-generation',
            daemon=True,
        )
        self._worker.start()

    def __iter__(self) -> Iterator[str]:
        while True:
            item = self._items.get()
            if item is _END:
                break
            yield item  # type: ignore[misc]
        if self._worker is not None:
            self._worker.join()
        if self._error is not None:
            raise self._error


__all__ = ['TextIteratorStreamer']

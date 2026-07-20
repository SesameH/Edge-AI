# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""A small chat-template facade attached to every native model handle."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modeling import _NativeModel


def _serialize_tools(tools: list[dict] | str | None) -> str | None:
    if tools is None:
        return None
    if isinstance(tools, str):
        if '\x00' in tools:
            raise ValueError('tools must not contain NUL bytes')
        return tools
    if isinstance(tools, list) and all(isinstance(tool, dict) for tool in tools):
        return json.dumps(tools)
    raise TypeError('tools must be a list of objects, a JSON string, or None')


class ChatTokenizer:
    """Expose a Transformers-shaped ``apply_chat_template`` method.

    UniRT performs tokenization inside the native backend; this facade only
    renders the chat string and therefore intentionally rejects
    ``tokenize=True``.
    """

    def __init__(self, model: '_NativeModel') -> None:
        self._model = model

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        tools: list[dict] | str | None = None,
    ) -> str:
        if not isinstance(tokenize, bool):
            raise TypeError('tokenize must be a boolean')
        if not isinstance(add_generation_prompt, bool):
            raise TypeError('add_generation_prompt must be a boolean')
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise TypeError('enable_thinking must be a boolean or None')
        if tokenize:
            raise ValueError(
                'UniRT never exposes token ids to Python: tokenization lives inside the '
                'native backend. Keep tokenize=False and hand the rendered string to '
                'model.generate().'
            )

        tools_json = _serialize_tools(tools)
        thinking = True if enable_thinking is None else enable_thinking
        if not thinking and not self._model.supports_thinking:
            warnings.warn(
                'ignoring enable_thinking=False: this model has no thinking mode, and '
                'forcing an empty <think></think> block tends to corrupt its replies.',
                stacklevel=2,
            )
            thinking = True

        return self._model._apply_chat_template(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=thinking,
            tools=tools_json,
        )


__all__ = ['ChatTokenizer']

# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Plugin capability flags, failure detail, and handle staleness."""

from __future__ import annotations

from ctypes import byref, c_void_p

import pytest

from conftest import model_path

NOT_INITIALIZED = -1005
INVALID_INPUT = -1001
PLUGIN_INVALID = -1211


def test_plugin_modalities_declared(sdk):
    plugins = set(sdk.get_runtime_list())
    assert 'llama_cpp' in plugins
    assert sdk.get_plugin_modalities('llama_cpp') == {'llm', 'vlm'}
    if 'mlx' in plugins:
        assert sdk.get_plugin_modalities('mlx') == {'llm'}
    if 'onnxruntime' in plugins:
        assert sdk.get_plugin_modalities('onnxruntime') == {'embedding'}


def test_modalities_of_unknown_plugin_raises_with_detail(sdk):
    from unirt._ffi._api import UniRTError

    with pytest.raises(UniRTError) as caught:
        sdk.get_plugin_modalities('no_such_backend')
    assert caught.value.code == PLUGIN_INVALID
    assert 'registered' in str(caught.value)


def test_last_error_message_reports_and_clears(sdk):
    lib = sdk.load_library()

    assert lib.unirt_llm_reset(None) == NOT_INITIALIZED
    assert b'NULL' in lib.unirt_last_error_message()

    assert lib.unirt_llm_reset(c_void_p(0xDEAD)) == INVALID_INPUT
    assert b'stale' in lib.unirt_last_error_message()

    # A successful shielded call clears the thread's detail.
    sdk.get_runtime_list()
    assert lib.unirt_last_error_message() == b''


def test_stale_handle_rejected_even_after_slot_reuse(sdk):
    """The ABA guard: a destroyed handle must stay dead even when a new
    model immediately reuses its registry slot (and possibly its memory)."""
    from unirt._ffi._types import unirt_LlmCreateInput

    lib = sdk.load_library()
    inp = unirt_LlmCreateInput(model_path=model_path('gguf').encode(), plugin_id=b'llama_cpp')

    first = c_void_p()
    assert lib.unirt_llm_create(byref(inp), byref(first)) == 0
    stale = c_void_p(first.value)
    assert lib.unirt_llm_destroy(first) == 0

    second = c_void_p()
    assert lib.unirt_llm_create(byref(inp), byref(second)) == 0
    try:
        assert stale.value != second.value
        assert lib.unirt_llm_reset(stale) == INVALID_INPUT
        assert lib.unirt_llm_destroy(stale) == INVALID_INPUT
        assert lib.unirt_llm_reset(second) == 0
    finally:
        assert lib.unirt_llm_destroy(second) == 0

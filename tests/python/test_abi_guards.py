# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ctypes import POINTER, byref, c_char_p, c_void_p, cast

from unirt._ffi._types import (
    unirt_EmbeddingEncodeOutput,
    unirt_GetDeviceListOutput,
    unirt_LlmCreateInput,
    unirt_LlmGenerateOutput,
    unirt_ResolveDeviceOutput,
)


INVALID_INPUT = -1001


def test_null_and_stale_llm_handles_are_rejected(sdk):
    lib = sdk.load_library()
    out_handle = c_void_p(123)
    assert lib.unirt_llm_create(None, byref(out_handle)) == INVALID_INPUT
    assert not out_handle.value

    output = unirt_LlmGenerateOutput()
    output.full_text = 123
    assert lib.unirt_llm_generate(None, None, byref(output)) == INVALID_INPUT
    assert not output.full_text
    assert lib.unirt_llm_destroy(c_void_p(1)) == INVALID_INPUT
    assert lib.unirt_llm_reset(c_void_p(1)) == INVALID_INPUT


def test_missing_plugin_id_is_rejected(sdk):
    lib = sdk.load_library()
    inp = unirt_LlmCreateInput()
    out_handle = c_void_p()
    assert lib.unirt_llm_create(byref(inp), byref(out_handle)) == INVALID_INPUT


def test_null_and_stale_embedding_handles_are_rejected(sdk):
    lib = sdk.load_library()
    out_handle = c_void_p(123)
    assert lib.unirt_embedding_create(None, byref(out_handle)) == INVALID_INPUT
    assert not out_handle.value

    output = unirt_EmbeddingEncodeOutput()
    output.embedding_count = 99
    output.embedding_dimension = 99
    assert lib.unirt_embedding_encode(None, None, byref(output)) == INVALID_INPUT
    assert not output.embeddings
    assert output.embedding_count == 0
    assert output.embedding_dimension == 0
    assert lib.unirt_embedding_destroy(c_void_p(1)) == INVALID_INPUT


def test_invalid_native_model_config_is_rejected_before_plugin_dispatch(sdk):
    lib = sdk.load_library()
    inp = unirt_LlmCreateInput(model_path=b'/does/not/exist', plugin_id=b'llama_cpp')
    inp.config.n_ctx = -1
    out_handle = c_void_p(123)
    assert lib.unirt_llm_create(byref(inp), byref(out_handle)) == INVALID_INPUT
    assert not out_handle.value


def test_plugin_enumeration_is_sorted(sdk):
    runtimes = sdk.get_runtime_list()
    assert runtimes == sorted(runtimes)


def test_failure_paths_zero_caller_owned_output_structs(sdk):
    lib = sdk.load_library()

    devices = unirt_GetDeviceListOutput()
    devices.device_ids = cast(c_void_p(123), POINTER(c_char_p))
    devices.device_names = cast(c_void_p(456), POINTER(c_char_p))
    devices.device_count = 99
    assert lib.unirt_get_device_list(None, byref(devices)) == INVALID_INPUT
    assert not devices.device_ids and not devices.device_names and devices.device_count == 0

    resolved = unirt_ResolveDeviceOutput(device_id=123, ngl=99, warning=456)
    assert lib.unirt_resolve_device(None, byref(resolved)) == INVALID_INPUT
    assert not resolved.device_id and not resolved.warning and resolved.ngl == 0


def test_positive_unknown_status_is_not_reported_as_success(sdk):
    assert sdk.load_library().unirt_get_error_message(1) == b'Unknown error code'

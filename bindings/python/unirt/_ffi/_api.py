# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Bound native entry points plus Python-owned runtime lifecycle."""

from __future__ import annotations

import atexit
import logging
import os
import threading
from ctypes import CFUNCTYPE, POINTER, byref, c_char_p, c_int32, c_uint32, c_void_p, string_at

from ._lib import load_library
from ._types import (
    unirt_EmbeddingCreateInput,
    unirt_EmbeddingEncodeInput,
    unirt_EmbeddingEncodeOutput,
    unirt_EmbeddingRuntimeStats,
    unirt_GetDeviceListInput,
    unirt_GetDeviceListOutput,
    unirt_GetPluginListOutput,
    unirt_KvCacheLoadInput,
    unirt_KvCacheLoadOutput,
    unirt_KvCacheSaveInput,
    unirt_KvCacheSaveOutput,
    unirt_LlmApplyChatTemplateInput,
    unirt_LlmApplyChatTemplateOutput,
    unirt_LlmCreateInput,
    unirt_LlmGenerateInput,
    unirt_LlmGenerateOutput,
    unirt_LlmModelInfo,
    unirt_LlmRuntimeStats,
    unirt_ResolveDeviceInput,
    unirt_ResolveDeviceOutput,
    unirt_VlmApplyChatTemplateInput,
    unirt_VlmApplyChatTemplateOutput,
    unirt_VlmCapabilities,
    unirt_VlmCreateInput,
    unirt_VlmGenerateInput,
    unirt_VlmGenerateOutput,
)

unirt_log_callback = CFUNCTYPE(None, c_int32, c_char_p)

_LEVEL_TRACE = 0
_LEVEL_DEBUG = 1
_LEVEL_INFO = 2
_LEVEL_WARN = 3
_LEVEL_ERROR = 4
_LEVEL_NONE = 5

_LOG_LEVELS = {
    'trace': logging.DEBUG,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'error': logging.ERROR,
    'none': logging.CRITICAL + 1,
}

UNIRT_ERROR_COMMON_BUSY = -1010
UNIRT_ERROR_COMMON_NOT_SUPPORTED = -1008
UNIRT_ERROR_COMMON_PLUGIN_INVALID = -1211
UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH = -1301

_logger = logging.getLogger('unirt')
_logger.addHandler(logging.NullHandler())

_state_lock = threading.RLock()
_bound = False
_initialized = False
_atexit_registered = False
_log_cb_ref: unirt_log_callback | None = None


class UniRTError(Exception):
    """Raised when a native UniRT operation returns a negative status."""

    def __init__(self, code: int, message: str):
        self.code = int(code)
        self.message = message
        super().__init__(f'UniRTError({self.code}): {message}')


_PROTOTYPES: dict[str, tuple[list[object], object]] = {
    'unirt_get_error_message': ([c_int32], c_char_p),
    'unirt_last_error_message': ([], c_char_p),
    'unirt_get_plugin_modalities': ([c_char_p, POINTER(c_uint32)], c_int32),
    'unirt_set_log': ([unirt_log_callback], c_int32),
    'unirt_init': ([], c_int32),
    'unirt_deinit': ([], c_int32),
    'unirt_free': ([c_void_p], None),
    'unirt_version': ([], c_char_p),
    'unirt_get_plugin_version': ([c_char_p], c_char_p),
    'unirt_get_plugin_list': ([POINTER(unirt_GetPluginListOutput)], c_int32),
    'unirt_get_device_list': (
        [POINTER(unirt_GetDeviceListInput), POINTER(unirt_GetDeviceListOutput)],
        c_int32,
    ),
    'unirt_resolve_device': (
        [POINTER(unirt_ResolveDeviceInput), POINTER(unirt_ResolveDeviceOutput)],
        c_int32,
    ),
    'unirt_llm_create': ([POINTER(unirt_LlmCreateInput), POINTER(c_void_p)], c_int32),
    'unirt_llm_destroy': ([c_void_p], c_int32),
    'unirt_llm_reset': ([c_void_p], c_int32),
    'unirt_llm_generate': (
        [c_void_p, POINTER(unirt_LlmGenerateInput), POINTER(unirt_LlmGenerateOutput)],
        c_int32,
    ),
    'unirt_llm_get_model_info': ([c_void_p, POINTER(unirt_LlmModelInfo)], c_int32),
    'unirt_llm_get_runtime_stats': (
        [c_void_p, POINTER(unirt_LlmRuntimeStats)],
        c_int32,
    ),
    'unirt_llm_apply_chat_template': (
        [
            c_void_p,
            POINTER(unirt_LlmApplyChatTemplateInput),
            POINTER(unirt_LlmApplyChatTemplateOutput),
        ],
        c_int32,
    ),
    'unirt_llm_save_kv_cache': (
        [c_void_p, POINTER(unirt_KvCacheSaveInput), POINTER(unirt_KvCacheSaveOutput)],
        c_int32,
    ),
    'unirt_llm_load_kv_cache': (
        [c_void_p, POINTER(unirt_KvCacheLoadInput), POINTER(unirt_KvCacheLoadOutput)],
        c_int32,
    ),
    'unirt_vlm_create': ([POINTER(unirt_VlmCreateInput), POINTER(c_void_p)], c_int32),
    'unirt_vlm_destroy': ([c_void_p], c_int32),
    'unirt_vlm_reset': ([c_void_p], c_int32),
    'unirt_vlm_generate': (
        [c_void_p, POINTER(unirt_VlmGenerateInput), POINTER(unirt_VlmGenerateOutput)],
        c_int32,
    ),
    'unirt_vlm_apply_chat_template': (
        [
            c_void_p,
            POINTER(unirt_VlmApplyChatTemplateInput),
            POINTER(unirt_VlmApplyChatTemplateOutput),
        ],
        c_int32,
    ),
    'unirt_vlm_get_capabilities': (
        [c_void_p, POINTER(unirt_VlmCapabilities)],
        c_int32,
    ),
    'unirt_embedding_create': (
        [POINTER(unirt_EmbeddingCreateInput), POINTER(c_void_p)],
        c_int32,
    ),
    'unirt_embedding_destroy': ([c_void_p], c_int32),
    'unirt_embedding_encode': (
        [c_void_p, POINTER(unirt_EmbeddingEncodeInput), POINTER(unirt_EmbeddingEncodeOutput)],
        c_int32,
    ),
    'unirt_embedding_get_runtime_stats': (
        [c_void_p, POINTER(unirt_EmbeddingRuntimeStats)],
        c_int32,
    ),
}


def _bind_all() -> None:
    library = load_library()
    for symbol, (arguments, result) in _PROTOTYPES.items():
        function = getattr(library, symbol)
        function.argtypes = arguments
        function.restype = result


def _sdk_log_bridge(level: int, raw_message: bytes | None) -> None:
    try:
        message = raw_message.decode('utf-8', errors='replace') if raw_message else ''
        if level == _LEVEL_ERROR:
            _logger.error(message)
        elif level == _LEVEL_WARN:
            _logger.warning(message)
        elif level == _LEVEL_INFO:
            _logger.info(message)
        else:
            _logger.debug(message)
    except BaseException:
        # Exceptions must never unwind through a C callback boundary.
        return


def _install_log_callback_locked() -> None:
    global _log_cb_ref
    if _log_cb_ref is not None:
        return
    callback = unirt_log_callback(_sdk_log_bridge)
    result = load_library().unirt_set_log(callback)
    if result < 0:
        _check(result)
    _log_cb_ref = callback

    requested = os.environ.get('UNIRT_LOG', '').strip().lower()
    if requested in {'off', '0', 'false'}:
        requested = 'none'
    if requested in _LOG_LEVELS:
        _logger.setLevel(_LOG_LEVELS[requested])


def _ensure_bound() -> None:
    global _bound
    with _state_lock:
        if _bound:
            return
        _bind_all()
        _bound = True
        _install_log_callback_locked()


def install_log_callback() -> None:
    _ensure_bound()
    with _state_lock:
        _install_log_callback_locked()


def set_log_level(level: str) -> None:
    if not isinstance(level, str):
        raise TypeError('log level must be a string')
    normalized = level.strip().lower()
    if normalized in _LOG_LEVELS:
        _logger.setLevel(_LOG_LEVELS[normalized])


def _check(code: int) -> None:
    if code >= 0:
        return
    library = load_library()
    raw = library.unirt_get_error_message(c_int32(code))
    message = raw.decode('utf-8', errors='replace') if raw else 'unknown error'
    try:
        detail_raw = library.unirt_last_error_message()
        detail = detail_raw.decode('utf-8', errors='replace') if detail_raw else ''
    except AttributeError:  # older native library without the symbol
        detail = ''
    if detail:
        message = f'{message} ({detail})'
    raise UniRTError(code, message)


def _encode(value: str | None) -> bytes | None:
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise TypeError('expected a string or None')
    if '\x00' in value:
        raise ValueError('strings passed to the native API cannot contain NUL bytes')
    return value.encode('utf-8')


def _str_list_to_c(strings: list[str]):
    if not isinstance(strings, list) or not all(isinstance(item, str) for item in strings):
        raise TypeError('expected a list of strings')
    if any('\x00' in item for item in strings):
        raise ValueError('strings passed to the native API cannot contain NUL bytes')
    if not strings:
        return None, 0
    array_type = c_char_p * len(strings)
    return array_type(*(item.encode('utf-8') for item in strings)), len(strings)


def _unknown_runtime_message(runtime: str, available: list[str]) -> str:
    choices = ', '.join(sorted(available))
    return f'Unknown runtime: {runtime}. Available runtimes: {choices}.'


def init() -> None:
    """Initialize plugin discovery once for this Python process."""
    global _initialized, _atexit_registered
    with _state_lock:
        if _initialized:
            return
        _ensure_bound()
        _check(load_library().unirt_init())
        _initialized = True
        if not _atexit_registered:
            atexit.register(_atexit_deinit)
            _atexit_registered = True


def deinit() -> None:
    """Deinitialize UniRT when no native model handles remain."""
    global _initialized
    with _state_lock:
        if not _initialized:
            return
        _check(load_library().unirt_deinit())
        _initialized = False


def _atexit_deinit() -> None:
    try:
        deinit()
    except Exception:
        # At interpreter shutdown it is safer to keep live plugin vtables
        # mapped than to surface BUSY or partially tear down dependencies.
        pass


def ensure_init() -> None:
    with _state_lock:
        needs_init = not _initialized
    if needs_init:
        init()


def _require_init(function_name: str) -> None:
    with _state_lock:
        initialized = _initialized
    if not initialized:
        raise RuntimeError(
            f'unirt.{function_name}() requires the runtime to be initialized. '
            'Call unirt.init() first.'
        )


def version() -> str:
    _ensure_bound()
    raw = load_library().unirt_version()
    return raw.decode('utf-8', errors='replace') if raw else ''


def get_plugin_version(plugin_id: str) -> str:
    ensure_init()
    if not isinstance(plugin_id, str) or not plugin_id or '\x00' in plugin_id:
        raise ValueError('plugin_id must be a non-empty NUL-free string')
    raw = load_library().unirt_get_plugin_version(plugin_id.encode('utf-8'))
    return raw.decode('utf-8', errors='replace') if raw else ''


MODALITY_LLM = 0x1
MODALITY_VLM = 0x2
MODALITY_EMBEDDING = 0x4


def get_plugin_modalities(plugin_id: str) -> set[str]:
    """Model kinds a plugin can host, e.g. ``{'llm', 'vlm'}``.

    An empty set means the plugin did not declare its capabilities.
    """
    ensure_init()
    if not isinstance(plugin_id, str) or not plugin_id or '\x00' in plugin_id:
        raise ValueError('plugin_id must be a non-empty NUL-free string')
    bits = c_uint32(0)
    _check(load_library().unirt_get_plugin_modalities(plugin_id.encode('utf-8'), byref(bits)))
    names = []
    if bits.value & MODALITY_LLM:
        names.append('llm')
    if bits.value & MODALITY_VLM:
        names.append('vlm')
    if bits.value & MODALITY_EMBEDDING:
        names.append('embedding')
    return set(names)


def get_runtime_list() -> list[str]:
    _require_init('get_runtime_list')
    _ensure_bound()
    library = load_library()
    output = unirt_GetPluginListOutput()
    _check(library.unirt_get_plugin_list(byref(output)))
    try:
        if output.plugin_count < 0:
            raise RuntimeError('native runtime returned a negative plugin count')
        return [
            output.plugin_ids[index].decode('utf-8', errors='replace')
            for index in range(output.plugin_count)
        ]
    finally:
        if output.plugin_ids:
            library.unirt_free(output.plugin_ids)


def get_compute_unit_list(runtime: str) -> list[tuple[str, str]]:
    _require_init('get_compute_unit_list')
    if not isinstance(runtime, str) or not runtime or '\x00' in runtime:
        raise ValueError('runtime must be a non-empty NUL-free string')
    available = get_runtime_list()
    if runtime not in available:
        raise UniRTError(
            UNIRT_ERROR_COMMON_PLUGIN_INVALID,
            _unknown_runtime_message(runtime, available),
        )

    library = load_library()
    input_value = unirt_GetDeviceListInput(plugin_id=runtime.encode('utf-8'))
    output = unirt_GetDeviceListOutput()
    _check(library.unirt_get_device_list(byref(input_value), byref(output)))
    try:
        if output.device_count < 0:
            raise RuntimeError('native runtime returned a negative device count')
        return [
            (
                output.device_ids[index].decode('utf-8', errors='replace'),
                output.device_names[index].decode('utf-8', errors='replace'),
            )
            for index in range(output.device_count)
        ]
    finally:
        if output.device_ids:
            library.unirt_free(output.device_ids)
        if output.device_names:
            library.unirt_free(output.device_names)


def _read_owned_string(pointer_value: int | None) -> str | None:
    if not pointer_value:
        return None
    return string_at(pointer_value).decode('utf-8', errors='replace')


def resolve_device(
    plugin_id: str,
    mode: str | None,
    ngl_default: int,
) -> tuple[str | None, int, str | None]:
    _ensure_bound()
    if not isinstance(plugin_id, str) or not plugin_id or '\x00' in plugin_id:
        raise ValueError('plugin_id must be a non-empty NUL-free string')
    if mode is not None and (not isinstance(mode, str) or '\x00' in mode):
        raise ValueError('mode must be a NUL-free string or None')
    if not isinstance(ngl_default, int) or isinstance(ngl_default, bool):
        raise TypeError('ngl_default must be an integer')
    if not -(2**31) <= ngl_default <= 2**31 - 1:
        raise ValueError('ngl_default must fit in int32')

    library = load_library()
    input_value = unirt_ResolveDeviceInput(
        plugin_id=plugin_id.encode('utf-8'),
        mode=_encode(mode),
        ngl_default=ngl_default,
    )
    output = unirt_ResolveDeviceOutput()
    _check(library.unirt_resolve_device(byref(input_value), byref(output)))
    try:
        return (
            _read_owned_string(output.device_id),
            int(output.ngl),
            _read_owned_string(output.warning),
        )
    finally:
        if output.device_id:
            library.unirt_free(output.device_id)
        if output.warning:
            library.unirt_free(output.warning)


__all__ = [
    'UniRTError',
    'UNIRT_ERROR_COMMON_BUSY',
    'UNIRT_ERROR_COMMON_NOT_SUPPORTED',
    'UNIRT_ERROR_COMMON_PLUGIN_INVALID',
    'UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH',
    '_check',
    '_encode',
    '_str_list_to_c',
    'deinit',
    'ensure_init',
    'get_compute_unit_list',
    'get_plugin_version',
    'get_plugin_modalities',
    'get_runtime_list',
    'init',
    'install_log_callback',
    'load_library',
    'resolve_device',
    'set_log_level',
    'version',
]

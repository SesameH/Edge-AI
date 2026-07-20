# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""UniRT's stable Python surface."""

from . import model_manager
from ._ffi._api import (
    UniRTError,
    deinit,
    get_compute_unit_list,
    get_plugin_version,
    get_runtime_list,
    init,
    set_log_level,
    version,
)
from ._version import __version__
from .auto import (
    AutoModelForCausalLM,
    AutoModelForEmbedding,
    AutoModelForVision2Seq,
    load,
    resolve_device_map,
)
from .generation import GenerateOutput, GenerationProfile, TextIteratorStreamer
from .modeling import UniRTEmbedding, UniRTLLM, UniRTVLM

__all__ = [
    '__version__',
    'AutoModelForCausalLM',
    'AutoModelForEmbedding',
    'AutoModelForVision2Seq',
    'GenerateOutput',
    'GenerationProfile',
    'TextIteratorStreamer',
    'UniRTError',
    'UniRTEmbedding',
    'UniRTLLM',
    'UniRTVLM',
    'deinit',
    'get_compute_unit_list',
    'get_plugin_version',
    'get_runtime_list',
    'init',
    'load',
    'model_manager',
    'resolve_device_map',
    'set_log_level',
    'version',
]

# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact ``ctypes`` representation of the public structs in ``unirt.h``.

Do not add ``_pack_`` or reorder fields: native alignment is part of the C ABI.
Heap-owned ``char *`` outputs use ``c_void_p`` so callers can pass the original
address to ``unirt_free`` instead of losing it to ctypes' automatic decoding.
"""

from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_int64,
    c_void_p,
)


unirt_token_callback = CFUNCTYPE(c_bool, c_char_p, c_void_p)


class unirt_GetPluginListOutput(Structure):
    _fields_ = [
        ('plugin_ids', POINTER(c_char_p)),
        ('plugin_count', c_int32),
    ]


class unirt_GetDeviceListInput(Structure):
    _fields_ = [('plugin_id', c_char_p)]


class unirt_GetDeviceListOutput(Structure):
    _fields_ = [
        ('device_ids', POINTER(c_char_p)),
        ('device_names', POINTER(c_char_p)),
        ('device_count', c_int32),
    ]


class unirt_ResolveDeviceInput(Structure):
    _fields_ = [
        ('plugin_id', c_char_p),
        ('mode', c_char_p),
        ('ngl_default', c_int32),
    ]


class unirt_ResolveDeviceOutput(Structure):
    _fields_ = [
        ('device_id', c_void_p),
        ('ngl', c_int32),
        ('warning', c_void_p),
    ]


class unirt_ProfileData(Structure):
    _fields_ = [
        ('ttft', c_int64),
        ('prompt_time', c_int64),
        ('decode_time', c_int64),
        ('prompt_tokens', c_int64),
        ('generated_tokens', c_int64),
        ('audio_duration', c_int64),
        ('prefill_speed', c_double),
        ('decoding_speed', c_double),
        ('real_time_factor', c_double),
        ('stop_reason', c_char_p),
    ]


class unirt_SamplerConfig(Structure):
    _fields_ = [
        ('temperature', c_float),
        ('top_p', c_float),
        ('top_k', c_int32),
        ('min_p', c_float),
        ('repetition_penalty', c_float),
        ('presence_penalty', c_float),
        ('frequency_penalty', c_float),
        ('seed', c_int32),
        ('grammar_path', c_char_p),
        ('grammar_string', c_char_p),
        ('enable_json', c_bool),
    ]


class unirt_GenerationConfig(Structure):
    _fields_ = [
        ('max_tokens', c_int32),
        ('stop', POINTER(c_char_p)),
        ('stop_count', c_int32),
        ('n_past', c_int32),
        ('sampler_config', POINTER(unirt_SamplerConfig)),
        ('image_paths', POINTER(c_char_p)),
        ('image_count', c_int32),
        ('image_max_length', c_int32),
        ('audio_paths', POINTER(c_char_p)),
        ('audio_count', c_int32),
        ('sliding_window', c_bool),
        ('sliding_window_n_keep', c_int32),
    ]


class unirt_ModelConfig(Structure):
    _fields_ = [
        ('n_ctx', c_int32),
        ('n_threads', c_int32),
        ('n_threads_batch', c_int32),
        ('n_batch', c_int32),
        ('n_ubatch', c_int32),
        ('n_seq_max', c_int32),
        ('n_gpu_layers', c_int32),
        ('chat_template_path', c_char_p),
        ('chat_template_content', c_char_p),
        ('grammar_str', c_char_p),
    ]


class unirt_LlmCreateInput(Structure):
    _fields_ = [
        ('model_path', c_char_p),
        ('tokenizer_path', c_char_p),
        ('config', unirt_ModelConfig),
        ('plugin_id', c_char_p),
        ('device_id', c_char_p),
    ]


class unirt_KvCacheSaveInput(Structure):
    _fields_ = [('path', c_char_p)]


class unirt_KvCacheSaveOutput(Structure):
    _fields_ = [('reserved', c_void_p)]


class unirt_KvCacheLoadInput(Structure):
    _fields_ = [('path', c_char_p)]


class unirt_KvCacheLoadOutput(Structure):
    _fields_ = [('reserved', c_void_p)]


class unirt_LlmChatMessage(Structure):
    _fields_ = [
        ('role', c_char_p),
        ('content', c_char_p),
    ]


class unirt_LlmApplyChatTemplateInput(Structure):
    _fields_ = [
        ('messages', POINTER(unirt_LlmChatMessage)),
        ('message_count', c_int32),
        ('tools', c_char_p),
        ('enable_thinking', c_bool),
        ('add_generation_prompt', c_bool),
    ]


class unirt_LlmApplyChatTemplateOutput(Structure):
    _fields_ = [('formatted_text', c_void_p)]


class unirt_LlmGenerateInput(Structure):
    _fields_ = [
        ('prompt_utf8', c_char_p),
        ('config', POINTER(unirt_GenerationConfig)),
        ('on_token', unirt_token_callback),
        ('user_data', c_void_p),
        ('input_ids', POINTER(c_int32)),
        ('input_ids_count', c_int32),
    ]


class unirt_LlmGenerateOutput(Structure):
    _fields_ = [
        ('full_text', c_void_p),
        ('profile_data', unirt_ProfileData),
    ]


class unirt_LlmModelInfo(Structure):
    _fields_ = [
        ('vocab_size', c_int32),
        ('bos_token', c_int32),
        ('add_bos', c_int32),
        ('reserved0', c_int32),
    ]


class unirt_LlmRuntimeStats(Structure):
    _fields_ = [
        ('model_bytes', c_int64),
        ('kv_cache_bytes', c_int64),
        ('device_peak_bytes', c_int64),
        ('process_rss_bytes', c_int64),
        ('device_name', c_char_p),
    ]


class unirt_VlmContent(Structure):
    _fields_ = [
        ('type', c_char_p),
        ('text', c_char_p),
    ]


class unirt_VlmChatMessage(Structure):
    _fields_ = [
        ('role', c_char_p),
        ('contents', POINTER(unirt_VlmContent)),
        ('content_count', c_int64),
    ]


class unirt_VlmCreateInput(Structure):
    _fields_ = [
        ('model_path', c_char_p),
        ('mmproj_path', c_char_p),
        ('config', unirt_ModelConfig),
        ('plugin_id', c_char_p),
        ('device_id', c_char_p),
        ('tokenizer_path', c_char_p),
    ]


class unirt_VlmApplyChatTemplateInput(Structure):
    _fields_ = [
        ('messages', POINTER(unirt_VlmChatMessage)),
        ('message_count', c_int32),
        ('tools', c_char_p),
        ('enable_thinking', c_bool),
        ('grounding', c_bool),
    ]


class unirt_VlmApplyChatTemplateOutput(Structure):
    _fields_ = [('formatted_text', c_void_p)]


class unirt_VlmCapabilities(Structure):
    _fields_ = [
        ('supports_vision', c_bool),
        ('supports_audio', c_bool),
    ]


class unirt_VlmGenerateInput(Structure):
    _fields_ = [
        ('prompt_utf8', c_char_p),
        ('config', POINTER(unirt_GenerationConfig)),
        ('on_token', unirt_token_callback),
        ('user_data', c_void_p),
    ]


class unirt_VlmGenerateOutput(Structure):
    _fields_ = [
        ('full_text', c_void_p),
        ('profile_data', unirt_ProfileData),
    ]


class unirt_EmbeddingCreateInput(Structure):
    _fields_ = [
        ('model_path', c_char_p),
        ('plugin_id', c_char_p),
        ('device_id', c_char_p),
        ('pooling', c_int32),
        ('normalize', c_bool),
        ('output_name', c_char_p),
    ]


class unirt_EmbeddingEncodeInput(Structure):
    _fields_ = [
        ('input_ids', POINTER(c_int64)),
        ('attention_mask', POINTER(c_int64)),
        ('token_type_ids', POINTER(c_int64)),
        ('batch_size', c_int32),
        ('sequence_length', c_int32),
    ]


class unirt_EmbeddingEncodeOutput(Structure):
    _fields_ = [
        ('embeddings', POINTER(c_float)),
        ('embedding_count', c_int32),
        ('embedding_dimension', c_int32),
    ]


class unirt_EmbeddingRuntimeStats(Structure):
    _fields_ = [
        ('model_bytes', c_int64),
        ('device_peak_bytes', c_int64),
        ('process_rss_bytes', c_int64),
        ('device_name', c_char_p),
    ]


__all__ = [name for name in globals() if name.startswith('unirt_')]

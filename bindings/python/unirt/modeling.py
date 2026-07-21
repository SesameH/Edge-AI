# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""High-level owners for native generation and embedding handles.

Both model kinds share lifecycle, locking, streaming, output ownership, and
generation marshalling through :class:`_NativeModel`. Subclasses define only
their native struct types and modality-specific operations.
"""

from __future__ import annotations

import codecs
import math
import os
import threading
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_int64,
    c_void_p,
    cast,
    pointer,
    string_at,
)

from tokenizers import Tokenizer

from ._ffi._api import (
    UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH,
    _check,
    _str_list_to_c,
    load_library,
)
from ._ffi._types import (
    unirt_EmbeddingEncodeInput,
    unirt_EmbeddingEncodeOutput,
    unirt_EmbeddingRerankInput,
    unirt_EmbeddingRerankOutput,
    unirt_EmbeddingRuntimeStats,
    unirt_GenerationConfig,
    unirt_KvCacheLoadInput,
    unirt_KvCacheLoadOutput,
    unirt_KvCacheSaveInput,
    unirt_KvCacheSaveOutput,
    unirt_LlmApplyChatTemplateInput,
    unirt_LlmApplyChatTemplateOutput,
    unirt_LlmChatMessage,
    unirt_LlmGenerateInput,
    unirt_LlmGenerateOutput,
    unirt_LlmRuntimeStats,
    unirt_SamplerConfig,
    unirt_VlmApplyChatTemplateInput,
    unirt_VlmApplyChatTemplateOutput,
    unirt_VlmCapabilities,
    unirt_VlmChatMessage,
    unirt_VlmContent,
    unirt_VlmGenerateInput,
    unirt_VlmGenerateOutput,
    unirt_VlmRuntimeStats,
    unirt_token_callback,
)
from .generation.output import GenerateOutput, GenerationProfile
from .generation.streamer import TextIteratorStreamer
from .tokenizer import ChatTokenizer

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def _enc(value: str | None) -> bytes | None:
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise TypeError('expected a string or None')
    if '\x00' in value:
        raise ValueError('strings passed to the native API cannot contain NUL bytes')
    return value.encode('utf-8')


def _decode_utf8(pointer_value) -> str:
    """Decode an owned native string while discarding an incomplete tail."""
    if not pointer_value:
        return ''
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    return decoder.decode(string_at(pointer_value), final=False)


def _finite_number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('sampling parameters must be finite numbers')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('sampling parameters must be finite numbers')
    return result


def _build_sampler(
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    seed: int,
    grammar: str | None,
    json_mode: bool,
) -> unirt_SamplerConfig:
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError('top_k must be an integer')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError('seed must be an integer')
    if top_k > _INT32_MAX:
        raise ValueError('top_k exceeds the native int32 range')
    if not _INT32_MIN <= seed <= _INT32_MAX:
        raise ValueError('seed exceeds the native int32 range')
    if grammar is not None and (not isinstance(grammar, str) or '\x00' in grammar):
        raise ValueError('grammar must be a NUL-free string or None')
    if not isinstance(json_mode, bool):
        raise TypeError('json_mode must be a boolean')

    temperature = _finite_number('temperature', temperature)
    top_p = _finite_number('top_p', top_p)
    min_p = _finite_number('min_p', min_p)
    repetition_penalty = _finite_number('repetition_penalty', repetition_penalty)
    presence_penalty = _finite_number('presence_penalty', presence_penalty)
    frequency_penalty = _finite_number('frequency_penalty', frequency_penalty)

    if not 0.0 <= temperature <= 2.0:
        raise ValueError('temperature must be between 0 and 2')
    if not 0.0 <= top_p <= 1.0 or not 0.0 <= min_p <= 1.0:
        raise ValueError('top_p and min_p must be between 0 and 1')
    if top_k < 0 or repetition_penalty < 0:
        raise ValueError('top_k and repetition_penalty cannot be negative')
    if grammar and json_mode:
        raise ValueError('grammar and json_mode are mutually exclusive')

    return unirt_SamplerConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        seed=seed,
        grammar_string=_enc(grammar),
        enable_json=json_mode,
    )


def _build_gen_config(
    max_new_tokens: int,
    stop: list[str],
    sampler: unirt_SamplerConfig,
    images: list[str],
    audios: list[str],
    sliding_window: bool = False,
    sliding_window_n_keep: int = 0,
    n_past: int = 0,
) -> tuple[unirt_GenerationConfig, object, object, object]:
    stop_values, stop_count = _str_list_to_c(stop)
    image_values, image_count = _str_list_to_c(images)
    audio_values, audio_count = _str_list_to_c(audios)

    config = unirt_GenerationConfig(
        max_tokens=max_new_tokens,
        stop_count=stop_count,
        n_past=n_past,
        sampler_config=pointer(sampler),
        image_count=image_count,
        audio_count=audio_count,
        sliding_window=sliding_window,
        sliding_window_n_keep=sliding_window_n_keep,
    )
    if stop_values is not None:
        config.stop = cast(stop_values, POINTER(c_char_p))
    if image_values is not None:
        config.image_paths = cast(image_values, POINTER(c_char_p))
    if audio_values is not None:
        config.audio_paths = cast(audio_values, POINTER(c_char_p))
    return config, stop_values, image_values, audio_values


def _validate_common_generate_args(
    prompt: str,
    max_new_tokens: int,
    n_past: int,
    sliding_window: bool,
    sliding_window_n_keep: int,
    stream: bool,
    stop: list[str] | None,
    kwargs: dict,
) -> list[str]:
    if kwargs:
        raise TypeError(f"unknown generation arguments: {', '.join(sorted(kwargs))}")
    if not isinstance(prompt, str) or '\x00' in prompt:
        raise ValueError('prompt must be a NUL-free string')
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or not 0 < max_new_tokens <= _INT32_MAX
    ):
        raise ValueError('max_new_tokens must be a positive integer')
    if (
        not isinstance(n_past, int)
        or isinstance(n_past, bool)
        or not 0 <= n_past <= _INT32_MAX
    ):
        raise ValueError('n_past must be a non-negative integer')
    if (
        not isinstance(sliding_window_n_keep, int)
        or isinstance(sliding_window_n_keep, bool)
        or not 0 <= sliding_window_n_keep <= _INT32_MAX
    ):
        raise ValueError('sliding_window_n_keep must be a non-negative integer')
    if not isinstance(stream, bool):
        raise TypeError('stream must be a boolean')
    if not isinstance(sliding_window, bool):
        raise TypeError('sliding_window must be a boolean')
    if stop is None:
        return []
    if not isinstance(stop, list) or not all(
        isinstance(item, str) and '\x00' not in item for item in stop
    ):
        raise ValueError('stop must be a list of NUL-free strings')
    return stop


def _validated_path_list(values: list[str] | None, label: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list) or not all(
        isinstance(path, str) and '\x00' not in path for path in values
    ):
        raise ValueError(f'{label} must be a list of NUL-free paths')
    return values


def _apply_meta(profile: GenerationProfile, meta: dict | None) -> GenerationProfile:
    if meta is not None:
        profile.backend = meta.get('backend')
        profile.device = meta.get('device')
        profile.quant = meta.get('quant')
        profile.model_path = meta.get('model_path')
    return profile


class _NativeModel:
    """Shared owner and execution pipeline for a single native model handle."""

    _c_prefix = ''
    _GenerateInput: type
    _GenerateOutput: type

    def __init__(self, handle: c_void_p, meta: dict | None = None) -> None:
        self._handle = handle
        self._meta = meta
        self._op_lock = threading.RLock()
        self.tokenizer = ChatTokenizer(self)

    def _native(self, operation: str):
        return getattr(load_library(), f'unirt_{self._c_prefix}_{operation}')

    def _require_open(self) -> None:
        if not self._handle:
            raise RuntimeError('model handle is closed')

    @property
    def supports_thinking(self) -> bool:
        if self._meta is None:
            return True
        value = self._meta.get('supports_thinking')
        return True if value is None else bool(value)

    def __repr__(self) -> str:
        entries: list[str] = []
        if self._meta:
            for label, key in (
                ('model', 'model_name'),
                ('backend', 'backend'),
                ('device', 'device'),
                ('quant', 'quant'),
            ):
                value = self._meta.get(key)
                if value:
                    entries.append(f"  {label}='{value}'")
        name = type(self).__name__
        return f'{name}()' if not entries else f'{name}(\n{",\n".join(entries)},\n)'

    def _generate_locked(self, prompt: str, config, callback) -> GenerateOutput:
        library = load_library()
        input_value = self._GenerateInput(
            prompt_utf8=prompt.encode('utf-8'),
            config=pointer(config),
            on_token=callback,
            user_data=None,
        )
        output = self._GenerateOutput()
        status = self._native('generate')(
            self._handle,
            byref(input_value),
            byref(output),
        )
        try:
            text = _decode_utf8(output.full_text)
            profile = _apply_meta(GenerationProfile.from_c(output.profile_data), self._meta)
        finally:
            if output.full_text:
                library.unirt_free(output.full_text)

        if status == UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH:
            profile.stop_reason = 'context_length'
        else:
            _check(status)
        return GenerateOutput.from_raw(text, profile)

    def _generate_blocking(self, prompt: str, config, *keepalive) -> GenerateOutput:
        @unirt_token_callback
        def ignore_token(_token, _user_data):
            return True

        with self._op_lock:
            self._require_open()
            result = self._generate_locked(prompt, config, ignore_token)
        return result

    def _generate_stream(self, prompt: str, config, *keepalive) -> TextIteratorStreamer:
        streamer = TextIteratorStreamer()
        callback = streamer._make_callback()

        def run() -> GenerateOutput:
            pinned = (config, *keepalive)
            try:
                with self._op_lock:
                    self._require_open()
                    return self._generate_locked(prompt, config, callback)
            finally:
                del pinned

        streamer.start(run)
        return streamer

    def _dispatch_generation(
        self,
        prompt: str,
        config: unirt_GenerationConfig,
        sampler: unirt_SamplerConfig,
        arrays: list[object],
        stream: bool,
    ) -> GenerateOutput | TextIteratorStreamer:
        runner = self._generate_stream if stream else self._generate_blocking
        return runner(prompt, config, sampler, *arrays)

    def reset(self) -> None:
        with self._op_lock:
            self._require_open()
            _check(self._native('reset')(self._handle))

    def close(self) -> None:
        with self._op_lock:
            if not self._handle:
                return
            _check(self._native('destroy')(self._handle))
            self._handle = None  # type: ignore[assignment]

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _build_llm_messages(messages: list[dict]):
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise TypeError('messages must be a list of objects')
    native_messages: list[unirt_LlmChatMessage] = []
    for message in messages:
        role = message.get('role')
        content = message.get('content', '')
        if not isinstance(role, str) or '\x00' in role:
            raise ValueError('each message role must be a NUL-free string')
        if not isinstance(content, str) or '\x00' in content:
            raise ValueError('LLM message content must be a NUL-free string')
        native_messages.append(
            unirt_LlmChatMessage(
                role=role.encode('utf-8'),
                content=content.encode('utf-8'),
            )
        )
    array_type = unirt_LlmChatMessage * len(native_messages)
    return array_type(*native_messages), len(native_messages)


class UniRTLLM(_NativeModel):
    """Language-model handle returned by ``AutoModelForCausalLM``."""

    _c_prefix = 'llm'
    _GenerateInput = unirt_LlmGenerateInput
    _GenerateOutput = unirt_LlmGenerateOutput

    def _apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: str | None,
    ) -> str:
        native_messages, count = _build_llm_messages(messages)
        input_value = unirt_LlmApplyChatTemplateInput(
            messages=native_messages,
            message_count=count,
            tools=_enc(tools),
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
        )
        output = unirt_LlmApplyChatTemplateOutput()
        library = load_library()
        with self._op_lock:
            self._require_open()
            _check(
                library.unirt_llm_apply_chat_template(
                    self._handle,
                    byref(input_value),
                    byref(output),
                )
            )
            try:
                return _decode_utf8(output.formatted_text)
            finally:
                if output.formatted_text:
                    library.unirt_free(output.formatted_text)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = 0,
        stop: list[str] | None = None,
        grammar: str | None = None,
        json_mode: bool = False,
        stream: bool = False,
        sliding_window: bool = False,
        sliding_window_n_keep: int = 0,
        n_past: int = 0,
        **kwargs,
    ) -> GenerateOutput | TextIteratorStreamer:
        self._require_open()
        stops = _validate_common_generate_args(
            prompt,
            max_new_tokens,
            n_past,
            sliding_window,
            sliding_window_n_keep,
            stream,
            stop,
            kwargs,
        )
        sampler = _build_sampler(
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            seed,
            grammar,
            json_mode,
        )
        config, *arrays = _build_gen_config(
            max_new_tokens,
            stops,
            sampler,
            [],
            [],
            sliding_window,
            sliding_window_n_keep,
            n_past,
        )
        return self._dispatch_generation(prompt, config, sampler, arrays, stream)

    def runtime_stats(self) -> dict:
        output = unirt_LlmRuntimeStats()
        with self._op_lock:
            self._require_open()
            _check(
                load_library().unirt_llm_get_runtime_stats(
                    self._handle,
                    byref(output),
                )
            )
        return {
            'model_bytes': int(output.model_bytes),
            'kv_cache_bytes': int(output.kv_cache_bytes),
            'device_peak_bytes': int(output.device_peak_bytes),
            'process_rss_bytes': int(output.process_rss_bytes),
            'device_name': (
                output.device_name.decode('utf-8', errors='replace')
                if output.device_name else None
            ),
        }

    def _kv_operation(self, operation: str, path: str) -> None:
        if not isinstance(path, str) or not path or '\x00' in path:
            raise ValueError('path must be a non-empty NUL-free string')
        if operation == 'save_kv_cache':
            input_value = unirt_KvCacheSaveInput(path=path.encode('utf-8'))
            output = unirt_KvCacheSaveOutput()
            function = load_library().unirt_llm_save_kv_cache
        else:
            input_value = unirt_KvCacheLoadInput(path=path.encode('utf-8'))
            output = unirt_KvCacheLoadOutput()
            function = load_library().unirt_llm_load_kv_cache
        with self._op_lock:
            self._require_open()
            _check(function(self._handle, byref(input_value), byref(output)))

    def save_kv_cache(self, path: str) -> None:
        self._kv_operation('save_kv_cache', path)

    def load_kv_cache(self, path: str) -> None:
        self._kv_operation('load_kv_cache', path)


def _messages_have_modality(messages: list[dict], modality: str) -> bool:
    aliases = {
        'image': {'image', 'image_url'},
        'audio': {'audio', 'input_audio'},
    }.get(modality, {modality})
    return any(
        isinstance(content, list)
        and any(
            isinstance(block, dict) and block.get('type') in aliases
            for block in content
        )
        for content in (message.get('content') for message in messages)
    )


def _vlm_content_value(block: dict) -> str:
    for key in ('text', 'image', 'audio'):
        value = block.get(key)
        if value:
            return value
    return ''


def _build_vlm_messages(messages: list[dict]):
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise TypeError('messages must be a list of objects')

    native_messages: list[unirt_VlmChatMessage] = []
    content_arrays: list[object] = []
    for message in messages:
        role = message.get('role')
        if not isinstance(role, str) or '\x00' in role:
            raise ValueError('each message role must be a NUL-free string')
        content = message.get('content', '')
        blocks: list[dict]
        if isinstance(content, str):
            if '\x00' in content:
                raise ValueError('message text must not contain NUL bytes')
            blocks = [{'type': 'text', 'text': content}]
        elif isinstance(content, list):
            if not all(isinstance(block, dict) for block in content):
                raise TypeError('multimodal content blocks must be objects')
            blocks = content
        else:
            raise TypeError('message content must be a string or a list of content blocks')

        native_blocks: list[unirt_VlmContent] = []
        for block in blocks:
            content_type = block.get('type', 'text')
            if (
                not isinstance(content_type, str)
                or not content_type
                or '\x00' in content_type
            ):
                raise ValueError('content block type must be a non-empty NUL-free string')
            value = _vlm_content_value(block)
            if not isinstance(value, str) or '\x00' in value:
                raise ValueError('content block value must be a NUL-free string')
            native_blocks.append(
                unirt_VlmContent(
                    type=content_type.encode('utf-8'),
                    text=value.encode('utf-8'),
                )
            )
        content_type = unirt_VlmContent * len(native_blocks)
        content_array = content_type(*native_blocks)
        content_arrays.append(content_array)
        native_messages.append(
            unirt_VlmChatMessage(
                role=role.encode('utf-8'),
                contents=content_array,
                content_count=len(native_blocks),
            )
        )

    message_type = unirt_VlmChatMessage * len(native_messages)
    return message_type(*native_messages), len(native_messages), content_arrays


class UniRTVLM(_NativeModel):
    """Multimodal handle returned by ``AutoModelForVision2Seq``."""

    _c_prefix = 'vlm'
    _GenerateInput = unirt_VlmGenerateInput
    _GenerateOutput = unirt_VlmGenerateOutput

    def __init__(self, handle: c_void_p, meta: dict | None = None) -> None:
        super().__init__(handle, meta)
        self._template_media: dict[str, tuple[bool, bool]] = {}

    def _apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: str | None,
    ) -> str:
        if not add_generation_prompt:
            raise NotImplementedError(
                'the current VLM C ABI always formats a generation prompt'
            )
        native_messages, count, content_arrays = _build_vlm_messages(messages)
        input_value = unirt_VlmApplyChatTemplateInput(
            messages=native_messages,
            message_count=count,
            tools=_enc(tools),
            enable_thinking=enable_thinking,
        )
        output = unirt_VlmApplyChatTemplateOutput()
        library = load_library()
        with self._op_lock:
            self._require_open()
            _check(
                library.unirt_vlm_apply_chat_template(
                    self._handle,
                    byref(input_value),
                    byref(output),
                )
            )
            try:
                prompt = _decode_utf8(output.formatted_text)
            finally:
                if output.formatted_text:
                    library.unirt_free(output.formatted_text)
            self._template_media[prompt] = (
                _messages_have_modality(messages, 'image'),
                _messages_have_modality(messages, 'audio'),
            )
            while len(self._template_media) > 32:
                self._template_media.pop(next(iter(self._template_media)))
            _pinned = content_arrays
            del _pinned
            return prompt

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = 0,
        stop: list[str] | None = None,
        grammar: str | None = None,
        json_mode: bool = False,
        images: list[str] | None = None,
        audios: list[str] | None = None,
        stream: bool = False,
        sliding_window: bool = False,
        sliding_window_n_keep: int = 0,
        n_past: int = 0,
        **kwargs,
    ) -> GenerateOutput | TextIteratorStreamer:
        self._require_open()
        stops = _validate_common_generate_args(
            prompt,
            max_new_tokens,
            n_past,
            sliding_window,
            sliding_window_n_keep,
            stream,
            stop,
            kwargs,
        )
        image_paths = _validated_path_list(images, 'images')
        audio_paths = _validated_path_list(audios, 'audios')
        with self._op_lock:
            expected_image, expected_audio = self._template_media.get(
                prompt,
                (False, False),
            )
        if expected_image and not image_paths:
            raise ValueError(
                'the chat template mentions an image, yet no file was supplied — '
                'call generate(..., images=["/path/to/img"]).'
            )
        if expected_audio and not audio_paths:
            raise ValueError(
                'the chat template mentions audio, yet no file was supplied — '
                'call generate(..., audios=["/path/to/clip"]).'
            )
        for path in image_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Image file not found: {path}')
        for path in audio_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Audio file not found: {path}')

        sampler = _build_sampler(
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            seed,
            grammar,
            json_mode,
        )
        config, *arrays = _build_gen_config(
            max_new_tokens,
            stops,
            sampler,
            image_paths,
            audio_paths,
            sliding_window,
            sliding_window_n_keep,
            n_past,
        )
        return self._dispatch_generation(prompt, config, sampler, arrays, stream)

    def capabilities(self) -> dict[str, bool]:
        output = unirt_VlmCapabilities()
        with self._op_lock:
            self._require_open()
            _check(
                load_library().unirt_vlm_get_capabilities(
                    self._handle,
                    byref(output),
                )
            )
        return {
            'vision': bool(output.supports_vision),
            'audio': bool(output.supports_audio),
        }

    def runtime_stats(self) -> dict:
        output = unirt_VlmRuntimeStats()
        with self._op_lock:
            self._require_open()
            _check(
                load_library().unirt_vlm_get_runtime_stats(
                    self._handle,
                    byref(output),
                )
            )
        return {
            'model_bytes': int(output.model_bytes),
            'kv_cache_bytes': int(output.kv_cache_bytes),
            'device_peak_bytes': int(output.device_peak_bytes),
            'process_rss_bytes': int(output.process_rss_bytes),
            'device_name': (
                output.device_name.decode('utf-8', errors='replace')
                if output.device_name else None
            ),
        }


class UniRTEmbedding:
    """ONNX encoder handle returned by ``AutoModelForEmbedding``."""

    def __init__(
        self,
        handle: c_void_p,
        tokenizer_path: str | None,
        *,
        max_length: int,
        padding_side: str = 'right',
        meta: dict | None = None,
    ) -> None:
        if not isinstance(max_length, int) or isinstance(max_length, bool) or not 0 < max_length <= _INT32_MAX:
            raise ValueError('max_length must be a positive int32 value')
        if padding_side not in {'left', 'right'}:
            raise ValueError("padding_side must be 'left' or 'right'")
        tokenizer = None
        if tokenizer_path is not None:
            try:
                tokenizer = Tokenizer.from_file(tokenizer_path)
            except Exception:
                # Do not leak a native model when tokenizer construction fails.
                load_library().unirt_embedding_destroy(handle)
                raise

        padding = tokenizer.padding or {} if tokenizer else {}
        pad_id = padding.get('pad_id')
        pad_token = padding.get('pad_token')
        if tokenizer and pad_id is None:
            for candidate in ('[PAD]', '<pad>', '<|pad|>'):
                candidate_id = tokenizer.token_to_id(candidate)
                if candidate_id is not None:
                    pad_id = candidate_id
                    pad_token = candidate
                    break

        self._handle = handle
        # None for rerank()-only use (unirt_embedding_rerank tokenizes
        # natively via the GGUF's own vocab); encode()/__call__ require one.
        self._tokenizer = tokenizer
        self._pad_id = int(pad_id) if pad_id is not None else None
        self._pad_token = pad_token
        self._max_length = max_length
        self._padding_side = padding.get('direction', padding_side).casefold()
        if self._padding_side not in {'left', 'right'}:
            self._padding_side = padding_side
        self._meta = meta
        self._op_lock = threading.RLock()

    def _require_open(self) -> None:
        if not self._handle:
            raise RuntimeError('model handle is closed')

    def __repr__(self) -> str:
        details = []
        if self._meta:
            for key in ('model_name', 'backend', 'device'):
                value = self._meta.get(key)
                if value:
                    details.append(f"  {key}='{value}'")
        return 'UniRTEmbedding()' if not details else f'UniRTEmbedding(\n{",\n".join(details)},\n)'

    @staticmethod
    def _validate_texts(texts: str | list[str]) -> tuple[list[str], bool]:
        scalar = isinstance(texts, str)
        values = [texts] if scalar else texts
        if not isinstance(values, list) or not values:
            raise ValueError('texts must be a string or a non-empty list of strings')
        if not all(isinstance(text, str) and '\x00' not in text for text in values):
            raise ValueError('texts must contain only NUL-free strings')
        if len(values) > _INT32_MAX:
            raise ValueError('text batch exceeds the native int32 range')
        return values, scalar

    def _tokenize(self, texts: list[str]) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        if self._tokenizer is None:
            raise RuntimeError(
                'this model was opened without a tokenizer (rerank()-only); encode() needs one'
            )
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(max_length=self._max_length)
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=True)
        if not encodings or any(not encoding.ids for encoding in encodings):
            raise ValueError('tokenizer produced an empty sequence')
        width = max(len(encoding.ids) for encoding in encodings)
        if width > _INT32_MAX:
            raise ValueError('tokenized sequence exceeds the native int32 range')
        if any(len(encoding.ids) != width for encoding in encodings) and self._pad_id is None:
            raise ValueError(
                'tokenizer has no padding token but the batch contains different sequence lengths'
            )

        ids: list[list[int]] = []
        masks: list[list[int]] = []
        type_ids: list[list[int]] = []
        for encoding in encodings:
            row_ids = list(encoding.ids)
            row_types = list(encoding.type_ids or [0] * len(row_ids))
            pad = width - len(row_ids)
            pad_ids = [self._pad_id or 0] * pad
            pad_mask = [0] * pad
            pad_types = [0] * pad
            if self._padding_side == 'left':
                ids.append(pad_ids + row_ids)
                masks.append(pad_mask + [1] * len(row_ids))
                type_ids.append(pad_types + row_types)
            else:
                ids.append(row_ids + pad_ids)
                masks.append([1] * len(row_ids) + pad_mask)
                type_ids.append(row_types + pad_types)
        return ids, masks, type_ids

    @staticmethod
    def _rectangular_int_batch(
        name: str,
        values: list[list[int]] | None,
        *,
        batch: int | None = None,
        width: int | None = None,
        binary: bool = False,
    ) -> tuple[list[int] | None, int, int]:
        if values is None:
            if batch is None or width is None:
                raise ValueError(f'{name} is required')
            return None, batch, width
        if not isinstance(values, list) or not values or not all(isinstance(row, list) for row in values):
            raise TypeError(f'{name} must be a non-empty list of integer lists')
        row_width = len(values[0])
        if row_width <= 0 or row_width > _INT32_MAX or len(values) > _INT32_MAX:
            raise ValueError(f'{name} has an invalid batch shape')
        if batch is not None and len(values) != batch:
            raise ValueError(f'{name} batch size does not match input_ids')
        if width is not None and row_width != width:
            raise ValueError(f'{name} sequence length does not match input_ids')
        flattened: list[int] = []
        for row in values:
            if len(row) != row_width:
                raise ValueError(f'{name} must be rectangular')
            for value in row:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f'{name} values must be integers')
                if not 0 <= value <= 2**63 - 1 or (binary and value not in {0, 1}):
                    raise ValueError(f'{name} contains an invalid value')
                flattened.append(value)
        return flattened, len(values), row_width

    def encode_tokens(
        self,
        input_ids: list[list[int]],
        *,
        attention_mask: list[list[int]] | None = None,
        token_type_ids: list[list[int]] | None = None,
    ) -> list[list[float]]:
        flat_ids, batch, width = self._rectangular_int_batch('input_ids', input_ids)
        flat_mask, _, _ = self._rectangular_int_batch(
            'attention_mask', attention_mask, batch=batch, width=width, binary=True
        )
        flat_types, _, _ = self._rectangular_int_batch(
            'token_type_ids', token_type_ids, batch=batch, width=width
        )
        assert flat_ids is not None

        ids_array = (c_int64 * len(flat_ids))(*flat_ids)
        mask_array = (c_int64 * len(flat_mask))(*flat_mask) if flat_mask is not None else None
        type_array = (c_int64 * len(flat_types))(*flat_types) if flat_types is not None else None
        input_value = unirt_EmbeddingEncodeInput(
            input_ids=ids_array,
            attention_mask=mask_array,
            token_type_ids=type_array,
            batch_size=batch,
            sequence_length=width,
        )
        output = unirt_EmbeddingEncodeOutput()
        library = load_library()
        with self._op_lock:
            self._require_open()
            status = library.unirt_embedding_encode(
                self._handle, byref(input_value), byref(output)
            )
            try:
                _check(status)
                if output.embedding_count != batch or output.embedding_dimension <= 0:
                    raise RuntimeError('native embedding output has an invalid shape')
                dimension = int(output.embedding_dimension)
                return [
                    [float(output.embeddings[row * dimension + column]) for column in range(dimension)]
                    for row in range(batch)
                ]
            finally:
                if output.embeddings:
                    library.unirt_free(cast(output.embeddings, c_void_p))

    def encode(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        values, scalar = self._validate_texts(texts)
        with self._op_lock:
            self._require_open()
            ids, masks, types = self._tokenize(values)
            embeddings = self.encode_tokens(
                ids,
                attention_mask=masks,
                token_type_ids=types,
            )
        return embeddings[0] if scalar else embeddings

    __call__ = encode

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score `documents` against `query` with the loaded model's
        classifier/rerank head — higher is more relevant. Unlike encode(),
        this takes raw text directly (the native side needs the model's own
        tokenizer and, when present, a model-specific rerank prompt
        template, neither reachable through the pre-tokenized encode() ABI).
        Raises UniRTError(PARAM_NOT_SUPPORTED) if the loaded model has no
        classifier head.
        """
        if not isinstance(query, str) or not query:
            raise ValueError('query must be a non-empty string')
        query_bytes = _enc(query)
        docs_array, doc_count = _str_list_to_c(documents)
        if doc_count == 0:
            raise ValueError('documents must be a non-empty list of strings')

        input_value = unirt_EmbeddingRerankInput(
            query_utf8=query_bytes,
            documents_utf8=docs_array,
            document_count=doc_count,
        )
        output = unirt_EmbeddingRerankOutput()
        library = load_library()
        with self._op_lock:
            self._require_open()
            status = library.unirt_embedding_rerank(self._handle, byref(input_value), byref(output))
            try:
                _check(status)
                if output.score_count != doc_count:
                    raise RuntimeError('native rerank output has an invalid shape')
                return [float(output.scores[i]) for i in range(doc_count)]
            finally:
                if output.scores:
                    library.unirt_free(cast(output.scores, c_void_p))

    def runtime_stats(self) -> dict:
        output = unirt_EmbeddingRuntimeStats()
        with self._op_lock:
            self._require_open()
            _check(
                load_library().unirt_embedding_get_runtime_stats(
                    self._handle, byref(output)
                )
            )
        return {
            'model_bytes': int(output.model_bytes),
            'device_peak_bytes': int(output.device_peak_bytes),
            'process_rss_bytes': int(output.process_rss_bytes),
            'device_name': (
                output.device_name.decode('utf-8', errors='replace')
                if output.device_name else None
            ),
        }

    def close(self) -> None:
        with self._op_lock:
            if not self._handle:
                return
            _check(load_library().unirt_embedding_destroy(self._handle))
            self._handle = None  # type: ignore[assignment]

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ['UniRTEmbedding', 'UniRTLLM', 'UniRTVLM']

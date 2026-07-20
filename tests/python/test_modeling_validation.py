# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ctypes import c_void_p, create_string_buffer

import pytest

from unirt.modeling import (
    UniRTLLM,
    UniRTVLM,
    _build_vlm_messages,
    _decode_utf8,
    _messages_have_modality,
)
from unirt.generation.streamer import TextIteratorStreamer


@pytest.fixture
def fake_model():
    model = UniRTLLM(c_void_p(1))
    yield model
    # This is not a native handle; prevent __del__ from calling destroy.
    model._handle = None


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'max_new_tokens': 0}, 'positive integer'),
        ({'temperature': float('nan')}, 'finite'),
        ({'top_p': 1.5}, 'between 0 and 1'),
        ({'top_k': -1}, 'cannot be negative'),
        ({'stop': ['bad\x00stop']}, 'NUL-free'),
        ({'grammar': 'root ::= "x"', 'json_mode': True}, 'mutually exclusive'),
        ({'n_past': -1}, 'non-negative'),
    ],
)
def test_generation_arguments_fail_before_native_call(fake_model, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fake_model.generate('prompt', **kwargs)


def test_unknown_generation_argument_is_not_silently_ignored(fake_model):
    with pytest.raises(TypeError, match='unknown generation arguments'):
        fake_model.generate('prompt', surprise=True)


def test_closed_model_rejects_operations(fake_model):
    fake_model._handle = None
    with pytest.raises(RuntimeError, match='closed'):
        fake_model.generate('prompt')


def test_integer_sampler_fields_reject_implicit_truncation(fake_model):
    with pytest.raises(TypeError, match='top_k must be an integer'):
        fake_model.generate('prompt', top_k=1.5)
    with pytest.raises(TypeError, match='seed must be an integer'):
        fake_model.generate('prompt', seed=1.5)


def test_falsey_wrong_types_are_not_normalized_away(fake_model):
    with pytest.raises(ValueError, match='stop must be a list'):
        fake_model.generate('prompt', stop='')
    with pytest.raises(TypeError, match='stream must be a boolean'):
        fake_model.generate('prompt', stream=0)
    with pytest.raises(ValueError, match='finite numbers'):
        fake_model.generate('prompt', temperature=True)


def test_vlm_message_validation_happens_before_native_call():
    with pytest.raises(TypeError, match='content blocks must be objects'):
        _build_vlm_messages([{'role': 'user', 'content': ['bad']}])
    with pytest.raises(ValueError, match='NUL-free'):
        _build_vlm_messages([{'role': 'user\x00', 'content': 'hello'}])


def test_closed_vlm_rejects_operations():
    model = UniRTVLM(c_void_p())
    with pytest.raises(RuntimeError, match='closed'):
        model.generate('prompt')


def test_vlm_template_rejects_unsupported_no_generation_prompt():
    model = UniRTVLM(c_void_p(1))
    try:
        with pytest.raises(NotImplementedError, match='always formats'):
            model._apply_chat_template([], False, False, None)
    finally:
        model._handle = None


def test_vlm_media_validation_covers_full_history_and_openai_aliases():
    messages = [
        {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {}}]},
        {'role': 'assistant', 'content': 'seen'},
        {'role': 'user', 'content': 'tell me more'},
    ]
    assert _messages_have_modality(messages, 'image')
    assert not _messages_have_modality(messages, 'audio')


def test_streamer_discards_partial_utf8_tail():
    streamer = TextIteratorStreamer()
    callback = streamer._make_callback()
    assert callback(b'ok', None)
    assert callback(b'\xe4', None)
    streamer.start(lambda: None)
    assert ''.join(streamer) == 'ok'


def test_blocking_decoder_discards_partial_utf8_tail():
    output = create_string_buffer(b'ok\xe4')
    assert _decode_utf8(output) == 'ok'


def test_tokenizer_facade_validates_control_arguments(fake_model):
    with pytest.raises(TypeError, match='enable_thinking'):
        fake_model.tokenizer.apply_chat_template([], enable_thinking=1)
    with pytest.raises(TypeError, match='tools must be'):
        fake_model.tokenizer.apply_chat_template([], tools={'type': 'function'})

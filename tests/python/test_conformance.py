# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""One contract, every text backend, checked side by side.

`unirt.h` is a promise that a caller can swap the backend under it. Nothing
was checking that: each backend had its own tests, each passed, and where they
disagreed nobody found out. The first run of this comparison turned up an
option that one backend honoured, the other refused, and *both* implemented
against the documented meaning (see the sliding-window tests below).

What can and cannot be asserted here:

- Not token equality. The two backends run different weights of the same model
  -- Q8_0 against fp16 -- so identical text is not something to demand, and a
  suite that demanded it would have to be switched off to stay green.
- The contract: which options are accepted, what errors invalid input
  produces, which properties hold of the output. Those are the same promise
  whatever the arithmetic underneath.

Genuine capability differences (no GBNF in MLX, no KV persistence in MLX) are
recorded in CAPABILITIES rather than skipped quietly, and the tests assert
both halves: that a supported feature works, and that an unsupported one fails
with the documented code instead of misbehaving.
"""

from __future__ import annotations

import json

import pytest
from conftest import model_path

from unirt._ffi._api import UniRTError

# Error codes the ABI defines for "this backend cannot do that", mirrored here
# from unirt.h the same way every other band is.
UNSUPPORTED_OPERATION = -1008
UNSUPPORTED_PARAMETER = -1009

# What each backend claims. A row that stops being true fails a test below --
# this is the matrix, not documentation of one.
CAPABILITIES = {
    'llama_cpp': {'gbnf_grammar': True, 'kv_cache_persistence': True},
    'mlx': {'gbnf_grammar': False, 'kv_cache_persistence': False},
}

MESSAGES = [{'role': 'user', 'content': 'Name one colour.'}]
SCHEMA = {'type': 'object', 'properties': {'a': {'type': 'string'}}, 'required': ['a']}


@pytest.fixture(params=['llama_cpp', 'mlx'])
def backend(request):
    return request.param


@pytest.fixture
def conforming(request, backend):
    """The same model, on whichever backend this round is about."""
    model = request.getfixturevalue(
        'llama_model' if backend == 'llama_cpp' else 'mlx_model'
    )
    model.reset()
    return model


def _prompt(model):
    return model._apply_chat_template(MESSAGES, True, False, None)


def _generate(model, **kwargs):
    kwargs.setdefault('max_new_tokens', 8)
    kwargs.setdefault('temperature', 0.0)
    return model.generate(_prompt(model), **kwargs)


# ---------------------------------------------------------------------------
# Behaviour every backend owes the caller
# ---------------------------------------------------------------------------


def test_the_chat_template_is_the_model_s_own(conforming):
    """Both backends read the template from the model, by different routes --
    llama.cpp from the GGUF metadata, MLX from tokenizer_config.json. Same
    model, so the rendered prompt has to come out the same, and a divergence
    here would silently change what every request asks."""
    prompt = _prompt(conforming)
    assert prompt == (
        '<|im_start|>user\nName one colour.<|im_end|>\n<|im_start|>assistant\n'
    )


def test_greedy_decoding_repeats_itself(conforming):
    first = _generate(conforming).text
    conforming.reset()
    assert _generate(conforming).text == first


def test_profile_reports_the_same_fields(conforming):
    profile = _generate(conforming).profile
    for field in ('prompt_tokens', 'generated_tokens', 'decode_speed', 'stop_reason'):
        assert getattr(profile, field), f'{field} was not reported'
    assert profile.generated_tokens <= 8


def test_streaming_and_blocking_agree(conforming):
    blocking = _generate(conforming, max_new_tokens=16).text
    conforming.reset()
    streamer = _generate(conforming, max_new_tokens=16, stream=True)
    assert ''.join(streamer) == blocking


def test_stop_sequences_stop_and_do_not_leak(conforming):
    text = _generate(conforming, max_new_tokens=24).text
    assert len(text) > 4
    stop = text[2:5]
    conforming.reset()
    out = _generate(conforming, max_new_tokens=24, stop=[stop])
    assert out.profile.stop_reason == 'stop_sequence'
    assert stop not in out.text


def test_a_json_schema_is_honoured(conforming):
    """Two entirely separate implementations -- llama.cpp compiles the schema
    to GBNF, the MLX plugin runs its own pushdown automaton over bytes -- and
    the caller is promised the same thing by both."""
    out = _generate(conforming, max_new_tokens=48, json_schema=SCHEMA)
    assert isinstance(json.loads(out.text), dict)
    assert 'a' in json.loads(out.text)


def test_json_mode_produces_json(conforming):
    out = _generate(conforming, max_new_tokens=48, json_mode=True)
    json.loads(out.text)


def test_logprobs_come_back_one_per_token(conforming):
    out = _generate(conforming, max_new_tokens=6, logprobs=3)
    assert out.logprobs is not None
    assert len(out.logprobs) == out.profile.generated_tokens
    for step in out.logprobs:
        assert step.chosen.logprob <= 0.0
        assert len(step.top) <= 3


# ---------------------------------------------------------------------------
# The same invalid input has to fail the same way
# ---------------------------------------------------------------------------


def test_an_out_of_range_n_past_is_rejected(conforming):
    with pytest.raises(UniRTError):
        _generate(conforming, n_past=2_000_000_000)


def test_an_empty_prompt_is_rejected(conforming):
    with pytest.raises(UniRTError):
        conforming.generate('', max_new_tokens=4, temperature=0.0)


def test_a_rejected_request_leaves_the_handle_usable(conforming):
    """Whatever the error was, it must not have left the KV and the plugin's
    transcript disagreeing -- the next request would then reuse a prefix that
    is not there."""
    before = _generate(conforming).text
    with pytest.raises(UniRTError):
        _generate(conforming, n_past=2_000_000_000)
    conforming.reset()
    assert _generate(conforming).text == before


# ---------------------------------------------------------------------------
# sliding_window: the divergence this suite was written after
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_context(request, backend):
    """A context small enough that a long answer must overflow it."""
    from unirt.auto import AutoModelForCausalLM

    if backend == 'mlx':
        from conftest import require_mlx

        require_mlx(request.getfixturevalue('sdk'))
    path = model_path('gguf' if backend == 'llama_cpp' else 'safetensors')
    model = AutoModelForCausalLM.from_pretrained(path, device_map=backend, n_ctx=64)
    yield model
    model.close()


def _overflowing(model, **kwargs):
    # Comfortably past the window even after llama.cpp rounds a small n_ctx up
    # -- it raised the 64 asked for above to 256, and a request that fits is
    # not a test of what happens when one does not.
    prompt = model._apply_chat_template(
        [{'role': 'user', 'content': 'Count from one to one hundred, one number per line.'}],
        True, False, None,
    )
    return model.generate(prompt, max_new_tokens=600, temperature=0.0, **kwargs)


def test_sliding_window_lets_generation_continue_past_the_window(tiny_context):
    out = _overflowing(tiny_context, sliding_window=True)
    assert out.text
    assert out.profile.stop_reason != 'context_length'


def test_without_sliding_window_overflow_is_reported(tiny_context):
    """The other half, and the one that was wrong: a backend must not decide
    on the caller's behalf that dropping the oldest tokens is acceptable."""
    assert _overflowing(tiny_context).profile.stop_reason == 'context_length'


# ---------------------------------------------------------------------------
# Capabilities: different on purpose, and still checked
# ---------------------------------------------------------------------------


def test_gbnf_grammar_matches_the_declared_capability(conforming, backend):
    supported = CAPABILITIES[backend]['gbnf_grammar']
    if supported:
        out = _generate(conforming, max_new_tokens=8, grammar='root ::= "yes"')
        assert out.text.startswith('yes')
        return
    with pytest.raises(UniRTError) as raised:
        _generate(conforming, max_new_tokens=8, grammar='root ::= "yes"')
    assert raised.value.code == UNSUPPORTED_PARAMETER


def test_kv_persistence_matches_the_declared_capability(conforming, backend, tmp_path):
    path = str(tmp_path / 'session.kv')
    supported = CAPABILITIES[backend]['kv_cache_persistence']
    if supported:
        first = _generate(conforming).text
        conforming.save_kv_cache(path)
        conforming.reset()
        conforming.load_kv_cache(path)
        # Restoring the session must restore what the prefix cache can reuse,
        # so the same request answers the same way.
        assert _generate(conforming).text == first
        return
    with pytest.raises(UniRTError) as raised:
        conforming.save_kv_cache(path)
    assert raised.value.code == UNSUPPORTED_OPERATION


def test_runtime_stats_report_the_same_shape(conforming):
    stats = conforming.runtime_stats()
    for field in ('model_bytes', 'device_name'):
        assert stats.get(field), f'{field} was not reported'
    # Present on both, and zero only before anything has been decoded.
    assert 'kv_cache_bytes' in stats
    _generate(conforming)
    assert conforming.runtime_stats()['kv_cache_bytes'] > 0

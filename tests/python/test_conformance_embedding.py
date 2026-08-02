# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""One encoder contract, every embedding backend, checked side by side.

The text backends could not be compared vector-for-vector -- different weights
of the same model produce different tokens, legitimately. Encoders can be:
the same sentence embedded by llama.cpp's GGUF path and by ONNX Runtime should
point in the same direction, whatever the arithmetic underneath, because that
direction is the entire output. So this suite does assert cross-backend
agreement, in the only unit that means anything for an embedding -- cosine.

It also runs the Core ML execution provider, which until this file existed had
never been executed by anything: the only test naming it asserted that the CLI
help text listed the option.
"""

from __future__ import annotations

import math

import pytest
from conftest import model_path

from unirt._ffi._api import UniRTError

UNSUPPORTED_PARAMETER = -1009

# Different on purpose: reranking needs a cross-encoder head, which the ONNX
# bundle for a bi-encoder does not have. Asserted both ways below.
CAPABILITIES = {
    'llama_cpp': {'rerank': True},
    'onnxruntime': {'rerank': False},
    'coreml': {'rerank': False},
}

TEXTS = [
    'a cat sat on the mat',
    'the feline rested on the rug',
    'quantum chromodynamics describes the strong interaction',
]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@pytest.fixture(params=['llama_cpp', 'onnxruntime', 'coreml'])
def encoder_name(request):
    return request.param


@pytest.fixture
def encoder(sdk, encoder_name):
    from unirt.auto import AutoModelForEmbedding

    if encoder_name == 'llama_cpp':
        path, device = model_path('encoder_gguf'), 'auto'
    else:
        if 'onnxruntime' not in sdk.get_runtime_list():
            pytest.skip('the ONNX Runtime plugin is not built in this configuration')
        path = model_path('encoder_onnx')
        device = 'cpu' if encoder_name == 'onnxruntime' else 'coreml'
    try:
        model = AutoModelForEmbedding.from_pretrained(path, device_map=device)
    except UniRTError as exc:
        if encoder_name == 'coreml' and exc.code == -1007:
            pytest.skip('Core ML is not available on this machine')
        raise
    yield model
    model.close()


# ---------------------------------------------------------------------------
# What every encoder owes the caller
# ---------------------------------------------------------------------------


def test_every_vector_has_the_models_width(encoder):
    vectors = encoder.encode(TEXTS)
    assert len({len(v) for v in vectors}) == 1
    assert len(vectors[0]) == 384


def test_vectors_come_back_normalized(encoder):
    for vector in encoder.encode(TEXTS):
        assert abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 1e-3


def test_encoding_is_repeatable(encoder):
    assert encoder.encode(TEXTS[0]) == encoder.encode(TEXTS[0])


def test_batching_does_not_change_the_direction(encoder):
    """Padding a batch changes the arithmetic slightly on some backends, which
    is fine -- what would not be fine is a vector that points somewhere else
    depending on what it was encoded alongside."""
    alone = encoder.encode(TEXTS[0])
    batched = encoder.encode(TEXTS)[0]
    assert _cosine(alone, batched) > 0.9999


def test_meaning_survives_the_encoder(encoder):
    """The property the whole thing exists for: paraphrases land near each
    other and unrelated text does not."""
    similar, paraphrase, unrelated = encoder.encode(TEXTS)
    assert _cosine(similar, paraphrase) > 0.4
    assert _cosine(similar, unrelated) < 0.2


def test_an_empty_batch_is_rejected(encoder):
    with pytest.raises(ValueError):
        encoder.encode([])


def test_rerank_matches_the_declared_capability(encoder, encoder_name):
    if CAPABILITIES[encoder_name]['rerank']:
        scores = encoder.rerank('a cat', TEXTS)
        assert len(scores) == len(TEXTS)
        # The paraphrase must beat the unrelated sentence, or the scores are
        # not ranking anything.
        assert scores[1] > scores[2]
        return
    with pytest.raises(UniRTError) as raised:
        encoder.rerank('a cat', TEXTS)
    assert raised.value.code == UNSUPPORTED_PARAMETER


# ---------------------------------------------------------------------------
# Across backends, not just within one
# ---------------------------------------------------------------------------


@pytest.fixture
def both_encoders(sdk):
    from unirt.auto import AutoModelForEmbedding

    if 'onnxruntime' not in sdk.get_runtime_list():
        pytest.skip('the ONNX Runtime plugin is not built in this configuration')
    gguf = AutoModelForEmbedding.from_pretrained(model_path('encoder_gguf'), device_map='auto')
    onnx = AutoModelForEmbedding.from_pretrained(model_path('encoder_onnx'), device_map='cpu')
    yield gguf, onnx
    gguf.close()
    onnx.close()


def test_the_two_backends_embed_the_same_sentence_the_same_way(both_encoders):
    """Same model, two entirely separate implementations -- llama.cpp's GGUF
    encoder against ONNX Runtime -- and one is 8-bit quantized. The vectors
    cannot be identical; pointing the same way is the actual contract."""
    gguf, onnx = both_encoders
    for text in TEXTS:
        agreement = _cosine(gguf.encode(text), onnx.encode(text))
        assert agreement > 0.98, f'{text!r} embedded differently: cosine {agreement:.3f}'


def test_the_two_backends_rank_a_corpus_the_same_way(both_encoders):
    """Absolute cosines drift with quantization; the order they induce is what
    a retrieval system actually uses, and that has to survive the backend."""
    gguf, onnx = both_encoders
    query = 'a cat sat on the mat'

    def ranking(model):
        vectors = model.encode(TEXTS)
        target = model.encode(query)
        return sorted(range(len(TEXTS)), key=lambda i: -_cosine(target, vectors[i]))

    assert ranking(gguf) == ranking(onnx)


def test_core_ml_agrees_with_the_cpu_provider(sdk):
    """Proof the Core ML path does more than load: ONNX Runtime places most of
    this graph on Core ML (373 of 418 nodes when this was written), and the
    result still has to match what the CPU provider computes."""
    from unirt.auto import AutoModelForEmbedding

    if 'onnxruntime' not in sdk.get_runtime_list():
        pytest.skip('the ONNX Runtime plugin is not built in this configuration')
    path = model_path('encoder_onnx')
    cpu = AutoModelForEmbedding.from_pretrained(path, device_map='cpu')
    try:
        try:
            accelerated = AutoModelForEmbedding.from_pretrained(path, device_map='coreml')
        except UniRTError as exc:
            if exc.code == -1007:
                pytest.skip('Core ML is not available on this machine')
            raise
        try:
            assert 'Core ML' in accelerated.runtime_stats()['device_name']
            for text in TEXTS:
                assert _cosine(cpu.encode(text), accelerated.encode(text)) > 0.999
        finally:
            accelerated.close()
    finally:
        cpu.close()

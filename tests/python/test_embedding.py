# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import pytest

from unirt import model_manager
from unirt.auto import (
    _embedding_precision,
    _embedding_max_length,
    _find_embedding_model,
)
from unirt.modeling import UniRTEmbedding


def test_embedding_precision_names_are_stable():
    assert _embedding_precision(None) == 'onnx'
    assert _embedding_precision('fp32') == 'onnx'
    assert _embedding_precision('qint8_arm64') == 'onnx-qint8-arm64'
    assert _embedding_precision('onnx-int8') == 'onnx-int8'


def test_model_manager_selects_onnx_without_mixing_safetensors():
    remote = model_manager._RemoteFile
    plan = model_manager._plan(
        'owner/encoder',
        [
            remote('model.safetensors', 10),
            remote('onnx/model.onnx', 20),
            remote('onnx/model_qint8_arm64.onnx', 8),
            remote('tokenizer.json', 3),
            remote('tokenizer_config.json', 4),
        ],
        'onnx-qint8-arm64',
    )
    assert plan.runtime == 'onnxruntime'
    assert plan.model_type == 'embedding'
    assert plan.precision == 'onnx-qint8-arm64'
    assert [item.name for item in plan.model_files] == ['onnx/model_qint8_arm64.onnx']
    assert plan.tokenizer and plan.tokenizer.name == 'tokenizer.json'


def test_mixed_format_manifest_keeps_per_variant_runtime(tmp_path):
    store = tmp_path / 'store'
    source = tmp_path / 'source'
    (source / 'onnx').mkdir(parents=True)
    (source / 'model.safetensors').write_bytes(b'safetensors')
    (source / 'onnx' / 'model.onnx').write_bytes(b'onnx')
    (source / 'tokenizer.json').write_text('{}', encoding='utf-8')

    model_manager.deinit()
    model_manager.init(str(store))
    try:
        model_manager.pull('owner/mixed', hub='localfs', local_path=str(source))
        assert model_manager.get_paths('owner/mixed:default').runtime == 'mlx'

        model_manager.pull(
            'owner/mixed', precision='onnx', hub='localfs', local_path=str(source)
        )
        assert model_manager.get_paths('owner/mixed:onnx').runtime == 'onnxruntime'
        assert model_manager.get_paths('owner/mixed:default').runtime == 'mlx'
    finally:
        model_manager.deinit()


def test_local_onnx_selection_is_deterministic(tmp_path):
    root = tmp_path
    (root / 'onnx').mkdir()
    (root / 'onnx' / 'model.onnx').write_bytes(b'canonical')
    (root / 'nested').mkdir()
    (root / 'nested' / 'model.onnx').write_bytes(b'deeper')
    selected = _find_embedding_model(str(root), 'onnx')
    assert selected == str(root / 'onnx' / 'model.onnx')


def test_embedding_max_length_prefers_smallest_valid_limit(tmp_path):
    (tmp_path / 'sentence_bert_config.json').write_text(
        '{"max_seq_length": 256}', encoding='utf-8'
    )
    (tmp_path / 'tokenizer_config.json').write_text(
        '{"model_max_length": 512}', encoding='utf-8'
    )
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{}', encoding='utf-8')
    assert _embedding_max_length(str(tmp_path), str(tokenizer), None) == 256
    assert _embedding_max_length(str(tmp_path), str(tokenizer), 128) == 128


@pytest.mark.parametrize(
    ('values', 'message'),
    [
        ([], 'non-empty'),
        ([[1], [1, 2]], 'rectangular'),
        ([[1, True]], 'integers'),
        ([[-1]], 'invalid value'),
    ],
)
def test_embedding_token_batch_validation(values, message):
    with pytest.raises((TypeError, ValueError), match=message):
        UniRTEmbedding._rectangular_int_batch('input_ids', values)


def test_real_embedding_model_when_explicitly_configured(sdk):
    bundle = os.environ.get('UNIRT_TEST_EMBEDDING_MODEL')
    if not bundle:
        pytest.skip('set UNIRT_TEST_EMBEDDING_MODEL for the real ONNX embedding smoke test')
    if 'onnxruntime' not in sdk.get_runtime_list():
        pytest.skip('ONNX Runtime plugin is unavailable')

    from unirt import AutoModelForEmbedding

    model = AutoModelForEmbedding.from_pretrained(bundle, device_map='cpu')
    try:
        vectors = model.encode(['a cat on a mat', 'a kitten on a rug'])
        assert len(vectors) == 2
        assert len(vectors[0]) == len(vectors[1]) > 0
        for vector in vectors:
            norm = sum(value * value for value in vector) ** 0.5
            assert norm == pytest.approx(1.0, abs=1e-5)
    finally:
        model.close()


def test_find_embedding_model_falls_back_to_gguf(tmp_path):
    (tmp_path / 'encoder-Q8_0.gguf').write_bytes(b'gguf')
    selected = _find_embedding_model(str(tmp_path), 'onnx')
    assert selected.endswith('encoder-Q8_0.gguf')
    assert _find_embedding_model(str(tmp_path), 'gguf') == selected


def test_find_embedding_model_prefers_onnx_when_both_exist(tmp_path):
    (tmp_path / 'model.onnx').write_bytes(b'onnx')
    (tmp_path / 'encoder.gguf').write_bytes(b'gguf')
    assert _find_embedding_model(str(tmp_path), 'onnx').endswith('model.onnx')
    assert _find_embedding_model(str(tmp_path), 'gguf').endswith('encoder.gguf')


def test_gguf_embedding_end_to_end(sdk):
    from conftest import REPO_ROOT

    bundle = os.path.join(REPO_ROOT, 'models', 'all-MiniLM-L6-v2-GGUF')
    if not os.path.isdir(bundle):
        pytest.skip('GGUF embedding bundle not downloaded (see README)')
    if 'llama_cpp' not in sdk.get_runtime_list():
        pytest.skip('llama_cpp plugin is unavailable')

    from unirt import AutoModelForEmbedding

    model = AutoModelForEmbedding.from_pretrained(bundle, device_map='cpu')
    try:
        vectors = model.encode([
            'a cat sits on the mat',
            'a kitten rests on the rug',
            'quarterly revenue rose sharply',
        ])
        assert len(vectors) == 3
        assert len(vectors[0]) == 384
        for vector in vectors:
            norm = sum(value * value for value in vector) ** 0.5
            assert norm == pytest.approx(1.0, abs=1e-4)

        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b))

        related = cosine(vectors[0], vectors[1])
        unrelated = cosine(vectors[0], vectors[2])
        assert related > unrelated + 0.3
    finally:
        model.close()

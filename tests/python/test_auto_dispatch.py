# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from unirt import auto
from unirt._ffi._api import UniRTError


def test_auto_dispatch_has_stable_priority(monkeypatch):
    monkeypatch.setattr(auto, 'get_runtime_list', lambda: ['mlx', 'llama_cpp'])
    monkeypatch.setattr(auto, 'resolve_device', lambda plugin, *_: (None, -1, None))
    assert auto.resolve_device_map('auto')[0] == 'llama_cpp'
    assert auto.resolve_device_map(' AUTO ')[0] == 'llama_cpp'


def test_model_format_binds_auto_backend(tmp_path, monkeypatch):
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'model.safetensors').write_bytes(b'x')
    assert auto._runtime_for_model_path(str(model_dir)) == 'mlx'
    assert auto._apply_plugin_hint('auto', 'mlx') == 'mlx'

    gguf = tmp_path / 'model.gguf'
    gguf.write_bytes(b'GGUF')
    assert auto._runtime_for_model_path(str(gguf)) == 'llama_cpp'


def test_local_directory_resolves_model_anchor_and_vlm_sidecars(tmp_path):
    model_dir = tmp_path / 'gguf-bundle'
    model_dir.mkdir()
    model = model_dir / 'tiny-Q4_0.gguf'
    projector = model_dir / 'mmproj-tiny-Q8_0.gguf'
    f16_projector = model_dir / 'mmproj-tiny-f16.gguf'
    tokenizer = model_dir / 'tokenizer.json'
    model.write_bytes(b'model')
    projector.write_bytes(b'projector')
    f16_projector.write_bytes(b'a-much-larger-f16-projector')
    tokenizer.write_text('{}', encoding='utf-8')

    resolved, mmproj, tok, paths = auto._resolve_model_sources(
        str(model_dir), None, None, None
    )
    assert resolved == str(model)
    assert mmproj == str(projector)
    assert tok == str(tokenizer)
    assert paths is None


def test_local_vlm_config_is_not_silently_loaded_as_text(tmp_path):
    model_dir = tmp_path / 'vlm'
    model_dir.mkdir()
    model = model_dir / 'model.safetensors'
    model.write_bytes(b'weights')
    (model_dir / 'config.json').write_text(
        '{"model_type":"mllama","vision_config":{}}', encoding='utf-8'
    )
    assert auto._is_vlm(None, 'uncached/model', str(model))


def test_explicit_incompatible_backend_is_rejected(tmp_path):
    gguf = tmp_path / 'model.gguf'
    gguf.write_bytes(b'GGUF')
    with pytest.raises(ValueError, match='cannot load this GGUF'):
        auto._validate_runtime_for_model('mlx', str(gguf))
    # Format inference describes bundled runtimes only; extensible plugins may
    # legitimately implement additional loaders.
    auto._validate_runtime_for_model('custom_runtime', str(gguf))


def test_bundled_vlm_capability_is_honest():
    assert auto._require_vlm_backend('llama_cpp') is None
    with pytest.raises(NotImplementedError, match='text-only'):
        auto._require_vlm_backend('mlx')


def test_llama_cpp_vlm_factory_reaches_native_plugin(sdk, tmp_path):
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'GGUF')
    with pytest.raises(UniRTError) as raised:
        auto.AutoModelForVision2Seq.from_pretrained(
            str(model),
            device_map='llama_cpp',
            mmproj_path=str(tmp_path / 'missing-mmproj.gguf'),
        )
    assert raised.value.code == -1004


def test_mlx_without_metal_is_rejected_before_model_allocation(monkeypatch):
    monkeypatch.setattr(auto, 'get_compute_unit_list', lambda _runtime: [])
    with pytest.raises(RuntimeError, match='no usable Apple Metal device'):
        auto._require_available_backend('mlx')

    # CPU-capable backends are not subject to the MLX Metal preflight.
    auto._require_available_backend('llama_cpp')
    with pytest.raises(RuntimeError, match='no UniRT inference backend'):
        auto._require_available_backend(None)


def test_model_config_rejects_lossy_or_unknown_values():
    with pytest.raises(TypeError, match='n_ctx must be an integer'):
        auto._build_model_config(1.5, -1)
    with pytest.raises(TypeError, match='n_threads must be an integer'):
        auto._build_model_config(0, -1, n_threads=2.5)
    with pytest.raises(TypeError, match='unknown model configuration argument'):
        auto._build_model_config(0, -1, typo_option=True)
    with pytest.raises(ValueError, match='NUL-free'):
        auto._build_model_config(0, -1, grammar_str='bad\x00grammar')


def test_dispatch_strings_reject_embedded_nul():
    with pytest.raises(ValueError, match='NUL-free'):
        auto.resolve_device_map('mlx\x00cpu')
    with pytest.raises(ValueError, match='NUL-free'):
        auto._resolve_model_sources('model\x00path', None, None, None)

# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from unirt import model_manager as mm
from unirt._ffi._api import UniRTError


@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    mm.deinit()
    mm.init(str(tmp_path / 'store'))
    yield
    mm.deinit()


def _install_fake_hub(monkeypatch, files: dict[str, bytes]):
    class FakeApi:
        def model_info(self, repo_id, *, files_metadata, token):
            assert repo_id == 'acme/tiny-GGUF'
            assert files_metadata is True
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename=name, size=len(content))
                    for name, content in files.items()
                ]
            )

    calls = []

    def fake_snapshot_download(*, repo_id, token, local_dir, allow_patterns, tqdm_class):
        calls.append((repo_id, tuple(sorted(allow_patterns))))
        for name in allow_patterns:
            target = Path(local_dir) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(files[name])
        return str(local_dir)

    monkeypatch.setattr(mm, 'HfApi', FakeApi)
    monkeypatch.setattr(mm, 'snapshot_download', fake_snapshot_download)
    return calls


def test_query_and_pull_only_selected_quant(monkeypatch):
    remote = {
        'tiny-Q4_0.gguf': b'four',
        'tiny-Q8_0-00001-of-00002.gguf': b'eight-a',
        'tiny-Q8_0-00002-of-00002.gguf': b'eight-b',
        'tokenizer.json': b'{}',
        'config.json': b'{"architectures":["LlamaForCausalLM"]}',
        'README.md': b'not required at runtime',
    }
    calls = _install_fake_hub(monkeypatch, remote)

    result = mm.query('acme/tiny-GGUF')
    assert result.runtime == 'llama_cpp'
    assert {item.precision for item in result.candidates} == {'Q4_0', 'Q8_0'}
    assert next(item.precision for item in result.candidates if item.is_default) == 'Q4_0'

    mm.pull('acme/tiny-GGUF', precision='Q8_0')
    paths = mm.get_paths('acme/tiny-GGUF:Q8_0')
    assert paths.model_path.endswith('tiny-Q8_0-00001-of-00002.gguf')
    assert paths.tokenizer_path and paths.tokenizer_path.endswith('tokenizer.json')
    assert not (Path(paths.model_dir) / 'tiny-Q4_0.gguf').exists()
    assert calls == [
        (
            'acme/tiny-GGUF',
            (
                'config.json',
                'tiny-Q8_0-00001-of-00002.gguf',
                'tiny-Q8_0-00002-of-00002.gguf',
                'tokenizer.json',
            ),
        )
    ]


def test_vlm_pull_prefers_q8_mmproj_without_downloading_f16(monkeypatch):
    remote = {
        'tiny-Q4_0.gguf': b'model',
        'mmproj-tiny-f16.gguf': b'f16-projector-is-larger',
        'mmproj-tiny-Q8_0.gguf': b'q8-projector',
        'config.json': b'{"vision_config":{}}',
    }
    calls = _install_fake_hub(monkeypatch, remote)

    query = mm.query('acme/tiny-GGUF')
    assert query.model_type == 'vlm'
    mm.pull('acme/tiny-GGUF', precision='Q4_0')
    paths = mm.get_paths('acme/tiny-GGUF:Q4_0')

    assert paths.mmproj_path and paths.mmproj_path.endswith('mmproj-tiny-Q8_0.gguf')
    assert calls == [
        (
            'acme/tiny-GGUF',
            ('config.json', 'mmproj-tiny-Q8_0.gguf', 'tiny-Q4_0.gguf'),
        )
    ]


def test_ensure_cached_reuses_manifest(monkeypatch):
    calls = _install_fake_hub(
        monkeypatch,
        {'tiny-Q4_0.gguf': b'model', 'tokenizer.json': b'{}'},
    )
    first = mm.ensure_cached('acme/tiny-GGUF')
    second = mm.ensure_cached('acme/tiny-GGUF')
    assert first == second
    assert len(calls) == 1


def test_local_safetensors_bundle_round_trip(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.safetensors').write_bytes(b'weights')
    (source / 'tokenizer.json').write_text('{}', encoding='utf-8')
    (source / 'config.json').write_text(
        json.dumps({'architectures': ['LlamaForCausalLM']}),
        encoding='utf-8',
    )

    mm.pull('acme/tiny-mlx', hub='localfs', local_path=str(source))
    paths = mm.get_paths('acme/tiny-mlx')
    assert paths.runtime == 'mlx'
    assert paths.model_type == 'llm'
    assert Path(paths.model_path).read_bytes() == b'weights'
    assert mm.list_models() == ['acme/tiny-mlx']
    assert mm.list_detailed()[0].total_size > 0

    mm.set_type('acme/tiny-mlx', 'vlm')
    assert mm.get_type('acme/tiny-mlx') == 'vlm'
    mm.remove('acme/tiny-mlx')
    assert mm.list_models() == []


def test_rejects_unsupported_hub():
    with pytest.raises(UniRTError) as raised:
        mm.pull('acme/tiny', hub='docker')
    assert raised.value.code == mm.UNIRT_ERROR_COMMON_NOT_SUPPORTED
    assert 'Hugging Face' in (mm.last_error_message() or '')


def test_clean_returns_removed_count(tmp_path):
    for repo in ('one', 'two'):
        source = tmp_path / repo
        source.mkdir()
        (source / 'model.safetensors').write_bytes(repo.encode())
        mm.pull(f'acme/{repo}', hub='localfs', local_path=str(source))
    assert mm.clean() == 2
    assert mm.list_models() == []


def test_sharded_safetensors_and_cache_integrity(tmp_path):
    source = tmp_path / 'sharded'
    source.mkdir()
    shard_1 = 'model-00001-of-00002.safetensors'
    shard_2 = 'model-00002-of-00002.safetensors'
    (source / shard_1).write_bytes(b'first-shard')
    (source / shard_2).write_bytes(b'second-shard')
    (source / 'adapter_model.safetensors').write_bytes(b'not-base-model')
    (source / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'a': shard_1, 'b': shard_2}}),
        encoding='utf-8',
    )
    (source / 'config.json').write_text(
        json.dumps({'model_type': 'llama', 'architectures': ['LlamaForCausalLM']}),
        encoding='utf-8',
    )
    (source / 'tokenizer.json').write_text('{}', encoding='utf-8')

    mm.pull('acme/sharded', hub='localfs', local_path=str(source))
    paths = mm.get_paths('acme/sharded')
    assert paths.model_path.endswith(shard_1)

    manifest = json.loads((Path(paths.model_dir) / 'unirt.json').read_text(encoding='utf-8'))
    record = manifest['ModelFile']['default']
    assert record['Files'] == [shard_1, shard_2]
    assert set(record['FileSizes']) == {shard_1, shard_2}
    assert 'adapter_model.safetensors' not in record['Files']

    # A cache entry is usable only while every recorded shard still matches
    # the atomic manifest's expected size.
    (Path(paths.model_dir) / shard_2).write_bytes(b'truncated')
    with pytest.raises(UniRTError) as raised:
        mm.get_paths('acme/sharded')
    assert raised.value.code == mm.UNIRT_ERROR_COMMON_FILE_NOT_FOUND
    assert mm.list_models() == []


def test_rejects_incomplete_numbered_shards(tmp_path):
    source = tmp_path / 'incomplete'
    source.mkdir()
    (source / 'model-00001-of-00002.safetensors').write_bytes(b'only-one')

    with pytest.raises(UniRTError) as raised:
        mm.pull('acme/incomplete', hub='localfs', local_path=str(source))
    assert raised.value.code == mm.UNIRT_ERROR_COMMON_MANIFEST_PARSE


def test_rejects_ambiguous_unnumbered_weight_files(tmp_path):
    source = tmp_path / 'ambiguous'
    source.mkdir()
    (source / 'one-Q4_0.gguf').write_bytes(b'one')
    (source / 'two-Q4_0.gguf').write_bytes(b'two')

    with pytest.raises(UniRTError, match='cannot be dispatched unambiguously'):
        mm.pull('acme/ambiguous', hub='localfs', local_path=str(source))


def test_manifest_sidecars_are_integrity_checked(tmp_path):
    source = tmp_path / 'sidecars'
    source.mkdir()
    (source / 'model.safetensors').write_bytes(b'weights')
    (source / 'config.json').write_text('{}', encoding='utf-8')

    mm.pull('acme/sidecars', hub='localfs', local_path=str(source))
    paths = mm.get_paths('acme/sidecars')
    (Path(paths.model_dir) / 'config.json').write_text('truncated', encoding='utf-8')
    with pytest.raises(UniRTError, match='wrong size'):
        mm.get_paths('acme/sidecars')


def test_tampered_manifest_cannot_delete_outside_cache(tmp_path):
    source = tmp_path / 'removal-source'
    source.mkdir()
    (source / 'tiny-Q4_0.gguf').write_bytes(b'model')
    mm.pull('acme/removal', hub='localfs', local_path=str(source))
    paths = mm.get_paths('acme/removal')

    victim = tmp_path / 'victim.txt'
    victim.write_text('keep me', encoding='utf-8')
    manifest_path = Path(paths.model_dir) / 'unirt.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['ModelFile']['Q8_0'] = {
        'Name': '../../../victim.txt',
        'Files': ['../../../victim.txt'],
        'Downloaded': True,
        'Size': victim.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(UniRTError, match='unsafe model file name'):
        mm.remove('acme/removal:Q8_0')
    assert victim.read_text(encoding='utf-8') == 'keep me'


def test_rejects_cross_platform_repository_path_tricks():
    for name in ('..\\escape/repo', 'acme/repo\\nested'):
        with pytest.raises(UniRTError):
            mm.query(name, hub='localfs', local_path='.')

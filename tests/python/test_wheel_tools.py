# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Platform-independent checks for the release wheel helper."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

from conftest import REPO_ROOT


def _load_helper():
    path = Path(REPO_ROOT) / 'bindings' / 'python' / 'build_wheel.py'
    spec = importlib.util.spec_from_file_location('unirt_build_wheel', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ('platform_tag', 'expected'),
    [
        ('win_amd64', ('unirt.dll', 'unirt_plugin.dll')),
        ('win_arm64', ('unirt.dll', 'unirt_plugin.dll')),
        ('macosx_14_0_arm64', ('libunirt.dylib', 'libunirt_plugin.dylib')),
        ('manylinux_2_31_x86_64', ('libunirt.so', 'libunirt_plugin.so')),
        ('musllinux_1_2_aarch64', ('libunirt.so', 'libunirt_plugin.so')),
        ('linux_x86_64', ('libunirt.so', 'libunirt_plugin.so')),
    ],
)
def test_native_names(platform_tag, expected):
    # Names come from the wheel's target tag, not the build host: the Windows
    # wheels are cross-built on macOS, where sys.platform would say 'darwin'.
    helper = _load_helper()
    assert helper._native_names(platform_tag) == expected


def test_native_names_rejects_unknown_tag():
    helper = _load_helper()
    with pytest.raises(ValueError, match='nonsense_tag'):
        helper._native_names('nonsense_tag')


def test_stamp_version(tmp_path):
    helper = _load_helper()
    version_file = tmp_path / '_version.py'
    version_file.write_text('VERSION = (0, 1, 0)\n', encoding='utf-8')

    helper._stamp_version(version_file, '2.3.4')

    assert version_file.read_text(encoding='utf-8') == 'VERSION = (2, 3, 4)\n'
    with pytest.raises(ValueError):
        helper._stamp_version(version_file, '2.3')


def test_verify_wheel_checks_native_layout(tmp_path):
    helper = _load_helper()
    tag = 'manylinux_2_28_x86_64'
    wheel = tmp_path / 'unirt-2.3.4-py3-none-manylinux_2_28_x86_64.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('unirt/_version.py', 'VERSION = (2, 3, 4)\n')
        archive.writestr('unirt/lib/libunirt.so', b'native')
        archive.writestr('unirt/lib/llama_cpp/libunirt_plugin.so', b'plugin')

    helper._verify_wheel(wheel, '2.3.4', tag)

    broken = tmp_path / 'broken.whl'
    with zipfile.ZipFile(broken, 'w') as archive:
        archive.writestr('unirt/_version.py', 'VERSION = (2, 3, 4)\n')
        archive.writestr('unirt/lib/libunirt.so', b'native')
    with pytest.raises(RuntimeError, match='libunirt_plugin.so'):
        helper._verify_wheel(broken, '2.3.4', tag)

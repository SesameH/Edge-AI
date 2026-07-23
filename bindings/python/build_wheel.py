#!/usr/bin/env python3
# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Build one platform wheel from an installed UniRT native prefix.

The Python binding uses ctypes rather than a CPython extension, so one
``py3-none-<platform>`` wheel serves every supported Python >= 3.10 on the
same OS/architecture. CMake installs the native runtime under ``lib/``;
this helper stages that tree inside ``unirt/lib``, stamps the release
version, builds the wheel, retags it, and verifies the resulting archive.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


_VERSION_RE = re.compile(r'^\d+\.\d+\.\d+$')
_VERSION_LINE_RE = re.compile(r'^VERSION = .*$', re.MULTILINE)


def _native_names() -> tuple[str, str]:
    if sys.platform == 'win32':
        return 'unirt.dll', 'unirt_plugin.dll'
    if sys.platform == 'darwin':
        return 'libunirt.dylib', 'libunirt_plugin.dylib'
    return 'libunirt.so', 'libunirt_plugin.so'


def _ignore_source(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        name
        for name in names
        if name in {'build', 'dist', '.venv', '__pycache__', 'lib'}
        or name.endswith('.egg-info')
        or name.endswith('.pyc')
    }
    return ignored


def _stamp_version(version_file: Path, version: str) -> None:
    if not _VERSION_RE.fullmatch(version):
        raise ValueError(f'version must be X.Y.Z, got {version!r}')
    parts = ', '.join(version.split('.'))
    source = version_file.read_text(encoding='utf-8')
    updated, count = _VERSION_LINE_RE.subn(f'VERSION = ({parts})', source)
    if count != 1:
        raise RuntimeError(f'expected one VERSION assignment in {version_file}')
    version_file.write_text(updated, encoding='utf-8')


def _verify_wheel(wheel_path: Path, version: str) -> None:
    main_library, plugin_library = _native_names()
    expected = {
        f'unirt/lib/{main_library}',
        f'unirt/lib/llama_cpp/{plugin_library}',
    }
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        missing = sorted(expected - names)
        if missing:
            raise RuntimeError(
                f'{wheel_path.name} is missing native files: {", ".join(missing)}'
            )
        version_source = archive.read('unirt/_version.py').decode('utf-8')
        expected_assignment = f"VERSION = ({', '.join(version.split('.'))})"
        if expected_assignment not in version_source:
            raise RuntimeError(
                f'{wheel_path.name} does not contain version {version}'
            )


def build_wheel(
    source_dir: Path,
    native_prefix: Path,
    version: str,
    platform_tag: str,
    output_dir: Path,
) -> Path:
    native_lib = native_prefix.resolve() / 'lib'
    main_library, plugin_library = _native_names()
    required = (
        native_lib / main_library,
        native_lib / 'llama_cpp' / plugin_library,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'native prefix is incomplete; missing: {", ".join(missing)}'
        )

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='unirt-wheel-') as temporary:
        staging = Path(temporary) / 'python'
        shutil.copytree(source_dir, staging, ignore=_ignore_source)
        shutil.copytree(native_lib, staging / 'unirt' / 'lib', symlinks=True)
        _stamp_version(staging / 'unirt' / '_version.py', version)

        subprocess.run(
            [sys.executable, '-m', 'build', '--wheel'],
            cwd=staging,
            check=True,
        )
        wheels = list((staging / 'dist').glob('*.whl'))
        if len(wheels) != 1:
            raise RuntimeError(f'expected one wheel before retagging, got {wheels}')

        subprocess.run(
            [
                sys.executable,
                '-m',
                'wheel',
                'tags',
                '--platform-tag',
                platform_tag,
                '--remove',
                str(wheels[0]),
            ],
            cwd=staging,
            check=True,
        )
        tagged = list((staging / 'dist').glob('*.whl'))
        if len(tagged) != 1:
            raise RuntimeError(f'expected one wheel after retagging, got {tagged}')

        destination = output_dir / tagged[0].name
        shutil.copy2(tagged[0], destination)

    _verify_wheel(destination, version)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=Path(__file__).resolve().parent,
        help='Python binding source (default: directory containing this script)',
    )
    parser.add_argument(
        '--native-prefix',
        required=True,
        type=Path,
        help='CMake install prefix containing lib/<native files>',
    )
    parser.add_argument('--version', required=True, help='Release version X.Y.Z')
    parser.add_argument(
        '--platform-tag',
        required=True,
        help='Wheel platform tag, for example win_amd64',
    )
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()

    wheel = build_wheel(
        args.source_dir,
        args.native_prefix,
        args.version,
        args.platform_tag,
        args.output_dir,
    )
    print(wheel)


if __name__ == '__main__':
    main()

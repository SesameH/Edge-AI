# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that a macOS wheel's binaries can run on the macOS its tag promises.

Nothing in the toolchain enforces this. `CMAKE_OSX_DEPLOYMENT_TARGET` defaults
to the *build machine's* OS, so when the runner image moved to macOS 26 every
dylib in the wheel started declaring `minos 26.4` while the wheel kept its
`macosx_14_0_arm64` tag. pip installs it happily on macOS 14 or 15; what breaks
is subtler than a load error -- llama.cpp still worked, because it compiles its
Metal shaders at runtime, while MLX quietly reported no devices, its Metal
library having been built for a newer system.

This is the macOS counterpart of the manylinux glibc check: the tag is a
promise, and something has to verify the binaries keep it.

    python tools/ci/check_macos_deployment_target.py dist/unirt-*-macosx_14_0_arm64.whl
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile
import zipfile

_TAG = re.compile(r'macosx_(\d+)_(\d+)_')
_MINOS = re.compile(r'^\s*minos\s+([\d.]+)\s*$', re.MULTILINE)


def tag_version(wheel_name: str) -> tuple[int, int]:
    match = _TAG.search(wheel_name)
    if not match:
        raise ValueError(f'not a macOS wheel name: {wheel_name}')
    return int(match.group(1)), int(match.group(2))


def parse_minos(otool_output: str) -> tuple[int, int] | None:
    """Return the LC_BUILD_VERSION minos as (major, minor), if the load command is there.

    A library built by an older toolchain carries LC_VERSION_MIN_MACOSX instead
    and has no `minos` line; those are older than anything this checks for.
    """

    match = _MINOS.search(otool_output)
    if not match:
        return None
    parts = match.group(1).split('.')
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def main(wheel: str) -> int:
    promised = tag_version(pathlib.Path(wheel).name)
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if n.endswith(('.dylib', '.so'))]
            archive.extractall(scratch, members=names)
        for name in sorted(names):
            path = pathlib.Path(scratch) / name
            output = subprocess.run(
                ['otool', '-l', str(path)], capture_output=True, text=True, check=False
            ).stdout
            minos = parse_minos(output)
            if minos is None:
                print(f'  {name}: no LC_BUILD_VERSION, nothing to check')
                continue
            verdict = 'ok' if minos <= promised else 'TOO NEW'
            print(f'  {name}: minos {minos[0]}.{minos[1]}  ({verdict})')
            if minos > promised:
                failures.append((name, minos))

    if failures:
        worst = max(m for _, m in failures)
        message = (
            f'{len(failures)} of the bundled libraries need macOS '
            f'{worst[0]}.{worst[1]}, but the wheel is tagged macosx_{promised[0]}_'
            f'{promised[1]}. Set CMAKE_OSX_DEPLOYMENT_TARGET to match the tag; '
            'it otherwise follows the build machine.'
        )
        print(f'::error::{message}')
        print(message, file=sys.stderr)
        return 1

    print(f'every library runs on macOS {promised[0]}.{promised[1]}, as the tag promises')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))

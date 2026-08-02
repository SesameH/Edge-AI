#!/usr/bin/env python3
# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Check an installed wheel dispatches to a CPU backend built for this machine.

A wheel has to run on every CPU of its architecture, so the one CPU backend it
would otherwise carry is built for the architecture's baseline -- SSE2 on
x86-64, plain armv8-a on aarch64. That costs real speed on every machine made
since, and it is invisible: inference works, it is simply slower than the same
hardware manages elsewhere.

The fix is one CPU backend per instruction-set level plus a load-time choice,
which fails in two directions this checks separately: the variants may not
have been built into the wheel at all, or they may all be there and the
loaded one still be the baseline.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import sys
from pathlib import Path

# What a backend must report to prove the choice was not the baseline. Both are
# a decade old in hardware terms -- AVX2 is Haswell (2013), DOTPROD is in every
# ARM server core and every Apple M-series -- so a runner that fails these is
# far more likely to mean broken dispatch than genuinely ancient hardware.
REQUIRED_FEATURE = {
    'x86_64': 'AVX2',
    'AMD64': 'AVX2',
    'aarch64': 'DOTPROD',
    'arm64': 'DOTPROD',
}


def _installed_backend_dir() -> Path:
    import unirt

    return Path(unirt.__file__).resolve().parent / 'lib' / 'llama_cpp'


def main() -> None:
    # With no argument this checks an installed wheel, so the source tree goes
    # off sys.path and the environment overrides go with it. Given one, it
    # checks that build tree instead -- which is how CI runs it against an
    # install prefix, before there is a wheel at all.
    explicit = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if explicit is None:
        source_dir = Path(__file__).resolve().parent
        sys.path[:] = [
            entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != source_dir
        ]
        os.environ.pop('UNIRT_LIB_PATH', None)
        os.environ.pop('UNIRT_PLUGIN_PATH', None)

    backend_dir = explicit if explicit is not None else _installed_backend_dir()
    variants = sorted(
        path.name for path in backend_dir.glob('*ggml-cpu-*') if path.is_file()
    )
    if len(variants) < 2:
        raise SystemExit(
            f'{backend_dir} carries {len(variants)} CPU backend module(s): '
            f'{variants or "none"}. This wheel was not built with '
            'UNIRT_LLAMA_CPU_VARIANTS, so every machine gets the baseline one.'
        )
    print(f'{len(variants)} CPU backends shipped: {", ".join(variants)}')

    # Which one ggml chose is reported by the plugin as it registers the
    # modules; that log line is the only place the choice is visible.
    records: list[str] = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger('unirt')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(Collector())

    from unirt._ffi import _api

    _api.init()
    try:
        # Enough to make the plugin open, which is where it registers ggml's
        # backend modules and reports what it got.
        _api.get_compute_unit_list('llama_cpp')
    finally:
        _api.deinit()

    registered = next((line for line in records if 'backends registered' in line), None)
    if registered is None:
        raise SystemExit(
            'the plugin never reported its ggml backends; either it did not '
            'load, or it was built without loadable backend modules'
        )
    features = re.search(r'CPU \(([^)]*)\)', registered)
    if not features:
        raise SystemExit(f'no CPU backend registered at all: {registered}')

    reported = features.group(1)
    required = REQUIRED_FEATURE.get(platform.machine())
    if required is None:
        print(f'no expectation for {platform.machine()}; CPU backend reports {reported}')
        return
    if f'{required}=1' not in reported:
        raise SystemExit(
            f'the CPU backend that loaded reports "{reported}", without {required}: '
            f'the baseline build was chosen over the {len(variants)} variants shipped '
            'beside it, so dispatch is not working'
        )
    print(f'dispatched to a CPU backend with {required}: {reported}')


if __name__ == '__main__':
    main()

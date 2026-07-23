#!/usr/bin/env python3
# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test an installed wheel without importing optional Python deps."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    source_dir = Path(__file__).resolve().parent
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or os.getcwd()).resolve() != source_dir
    ]
    spec = importlib.util.find_spec('unirt')
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError('the unirt package is not installed')
    package_root = Path(next(iter(spec.submodule_search_locations)))

    # Importing unirt normally also imports its Hugging Face helpers. Loading
    # _lib.py directly keeps this native-layout smoke independent of those
    # optional test-environment packages.
    os.environ.pop('UNIRT_LIB_PATH', None)
    os.environ.pop('UNIRT_PLUGIN_PATH', None)
    loader = runpy.run_path(str(package_root / '_ffi' / '_lib.py'))
    library = loader['load_library']()
    library.unirt_init.argtypes = []
    library.unirt_init.restype = ctypes.c_int32
    library.unirt_deinit.argtypes = []
    library.unirt_deinit.restype = ctypes.c_int32
    library.unirt_get_plugin_version.argtypes = [ctypes.c_char_p]
    library.unirt_get_plugin_version.restype = ctypes.c_char_p

    status = library.unirt_init()
    if status != 0:
        raise RuntimeError(f'unirt_init failed: {status}')
    try:
        plugin_version = library.unirt_get_plugin_version(b'llama_cpp')
        if not plugin_version:
            raise RuntimeError('the packaged llama_cpp plugin was not discovered')
        print(f'llama_cpp {plugin_version.decode("utf-8", errors="replace")}')
    finally:
        status = library.unirt_deinit()
        if status != 0:
            raise RuntimeError(f'unirt_deinit failed: {status}')


if __name__ == '__main__':
    main()

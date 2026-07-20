# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Thread-safe discovery and loading of the UniRT native library."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import threading

_logger = logging.getLogger('unirt')
_load_lock = threading.RLock()
_lib: ctypes.CDLL | None = None

# Keep these objects alive. Windows removes a DLL directory when its handle is
# closed, and dependency preloads must remain mapped while plugins are active.
_dll_directory_handles: list[object] = []
_preloaded_libraries: dict[str, ctypes.CDLL] = {}


def _lib_name() -> str:
    if sys.platform == 'win32':
        return 'unirt.dll'
    if sys.platform == 'darwin':
        return 'libunirt.dylib'
    return 'libunirt.so'


def _shared_lib_files(directory: str) -> list[str]:
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    if sys.platform == 'win32':
        return sorted(name for name in names if name.lower().endswith('.dll'))
    if sys.platform == 'darwin':
        return sorted(name for name in names if '.dylib' in name.lower())
    return sorted(
        name for name in names
        if '.so' in name.lower() and not name.lower().endswith('.a')
    )


def _preload_shared_libs(directories: list[str]) -> None:
    mode = getattr(ctypes, 'RTLD_LOCAL', 0)
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in _shared_lib_files(directory):
            path = os.path.join(directory, name)
            identity = os.path.realpath(path)
            if identity in _preloaded_libraries:
                continue
            try:
                _preloaded_libraries[identity] = ctypes.CDLL(path, mode=mode)
            except OSError:
                # A directory may contain optional or foreign-architecture
                # backends. The actual main-library load below remains strict.
                continue


def _dependency_directories(lib_path: str, extra_dirs: list[str] | None) -> list[str]:
    root = os.path.dirname(os.path.abspath(lib_path))
    directories = [root]
    try:
        directories.extend(
            entry.path for entry in os.scandir(root) if entry.is_dir()
        )
    except OSError:
        pass
    if extra_dirs:
        directories.extend(extra_dirs)

    unique: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        normalized = os.path.abspath(directory)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _setup_env(
    lib_path: str,
    plugin_path: str,
    extra_dirs: list[str] | None = None,
) -> None:
    directories = _dependency_directories(lib_path, extra_dirs)
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        for directory in directories:
            if os.path.isdir(directory):
                _dll_directory_handles.append(os.add_dll_directory(directory))

    _preload_shared_libs(directories)

    if sys.platform == 'win32':
        search_var = 'PATH'
    elif sys.platform == 'darwin':
        search_var = 'DYLD_LIBRARY_PATH'
    else:
        search_var = 'LD_LIBRARY_PATH'
    current = os.environ.get(search_var)
    prefix = os.pathsep.join(directories)
    os.environ[search_var] = prefix if not current else f'{prefix}{os.pathsep}{current}'
    os.environ.setdefault('UNIRT_PLUGIN_PATH', plugin_path)


def _find_release(name: str) -> tuple[str, str] | None:
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_dir = os.path.join(package_root, 'lib')
    candidate = os.path.join(lib_dir, name)
    return (candidate, lib_dir) if os.path.isfile(candidate) else None


def _find_dev(name: str) -> tuple[str, str] | None:
    cursor = os.path.dirname(os.path.abspath(__file__))
    for _depth in range(10):
        if os.path.isdir(os.path.join(cursor, 'sdk')):
            lib_dir = os.path.join(cursor, 'sdk', 'pkg-unirt', 'lib')
            candidate = os.path.join(lib_dir, name)
            return (candidate, lib_dir) if os.path.isfile(candidate) else None
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        cursor = parent
    return None


def _pick_layout(name: str) -> tuple[str, str] | None:
    release = _find_release(name)
    development = _find_dev(name)
    if release is None or development is None:
        return release or development

    winner = max((release, development), key=lambda item: os.path.getmtime(item[0]))
    label = 'release' if winner == release else 'dev'
    _logger.warning(
        'Both release (%s) and dev (%s) native libraries exist; using %s (newer).',
        release[0],
        development[0],
        label,
    )
    return winner


def _load_path(path: str, plugin_dir: str) -> ctypes.CDLL:
    _setup_env(path, plugin_dir)
    return ctypes.CDLL(path)


def load_library() -> ctypes.CDLL:
    """Return the process-wide native library handle, loading it once."""
    global _lib
    with _load_lock:
        if _lib is not None:
            return _lib

        name = _lib_name()
        override = os.environ.get('UNIRT_LIB_PATH')
        if override:
            candidate = os.path.abspath(override)
            if os.path.isdir(candidate):
                candidate = os.path.join(candidate, name)
            if not os.path.isfile(candidate):
                raise OSError(f'UNIRT_LIB_PATH does not contain {name}: {override}')
            _lib = _load_path(candidate, os.path.dirname(candidate))
            return _lib

        layout = _pick_layout(name)
        if layout is not None:
            path, plugin_dir = layout
            _lib = _load_path(path, plugin_dir)
            return _lib

        linker_name = ctypes.util.find_library('unirt')
        if linker_name:
            try:
                _lib = ctypes.CDLL(linker_name)
                return _lib
            except OSError:
                pass

        raise OSError(
            f'Cannot find the unirt native library ({name}).\n\n'
            'Build and install the SDK, or set UNIRT_LIB_PATH to the library '
            'file or its containing directory.'
        )


__all__ = ['load_library']

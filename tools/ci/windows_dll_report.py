# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Report which installed Windows library fails to load, and why.

`LoadLibraryExW failed: 126` (ERROR_MOD_NOT_FOUND) says a dependency could not
be resolved, but not which one -- and the missing module may be a dependency of
a dependency. Loading each library on its own, under the same search flags the
plugin loader uses, names the first that cannot come up.

    python tools/ci/windows_dll_report.py sdk/pkg-unirt/lib
"""

from __future__ import annotations

import ctypes
import os
import sys

# The flags registry.cpp passes to LoadLibraryExW. Deliberately no
# LOAD_WITH_ALTERED_SEARCH_PATH and no PATH: a dependency reachable only
# through PATH is exactly the kind of failure this is looking for.
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x100
LOAD_LIBRARY_SEARCH_APPLICATION_DIR = 0x200
LOAD_LIBRARY_SEARCH_USER_DIRS = 0x400
LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x800
FLAGS = (
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
    | LOAD_LIBRARY_SEARCH_APPLICATION_DIR
    | LOAD_LIBRARY_SEARCH_USER_DIRS
    | LOAD_LIBRARY_SEARCH_SYSTEM32
)

# Runtime DLLs worth reporting on directly: a missing one of these is a
# packaging problem rather than a build problem.
RUNTIME_CANDIDATES = (
    'vcomp140.dll',
    'vcomp140d.dll',
    'MSVCP140.dll',
    'VCRUNTIME140.dll',
    'VCRUNTIME140_1.dll',
    'libomp.dll',
)


def main(root: str) -> int:
    if os.name != 'nt':
        print('windows only', file=sys.stderr)
        return 0

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.LoadLibraryExW.restype = ctypes.c_void_p
    kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]

    failures = 0
    for directory, _, names in os.walk(root):
        for name in sorted(names):
            if not name.lower().endswith('.dll'):
                continue
            path = os.path.abspath(os.path.join(directory, name))
            handle = kernel32.LoadLibraryExW(path, None, FLAGS)
            if handle:
                print(f'loaded     {name}')
            else:
                failures += 1
                print(f'LOAD FAIL  {name}  error {ctypes.get_last_error()}')

    system32 = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'System32')
    for name in RUNTIME_CANDIDATES:
        print(f'System32 has {name}: {os.path.isfile(os.path.join(system32, name))}')

    # Never fail the step: this is a diagnostic, and the suite that follows is
    # what decides whether the job passes.
    print(f'{failures} of the installed libraries could not be loaded')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else 'sdk/pkg-unirt/lib'))

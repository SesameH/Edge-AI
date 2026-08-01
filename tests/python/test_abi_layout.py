# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""The C header and the ctypes mirror must describe the same bytes.

`unirt.h` and `bindings/python/unirt/_ffi/_types.py` are two hand-written
descriptions of one memory layout. Nothing enforced that they agree: a field
added to one and forgotten in the other, or added where padding falls
differently, compiles cleanly, imports cleanly, and then reads the wrong
offset -- a whole struct's worth of garbage crossing the boundary with no
error anywhere.

The C side of this is `tests/native/test_abi_layout.cpp`, built by CMake as
`unirt-abi-layout-test`. It prints its offsets as JSON; this compares them
field by field with what ctypes computed for the same struct.
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import subprocess

import pytest

from conftest import REPO_ROOT

from unirt._ffi import _types


def _layout_binary() -> pathlib.Path:
    root = pathlib.Path(REPO_ROOT)
    candidates = [
        root / 'build' / 'unirt-abi-layout-test',
        root / 'build' / 'Release' / 'unirt-abi-layout-test.exe',
        root / 'build' / 'unirt-abi-layout-test.exe',
    ]
    for path in candidates:
        if path.is_file():
            return path
    pytest.skip(
        'unirt-abi-layout-test is not built; configure with UNIRT_NATIVE_TESTS=ON '
        'and build before running this'
    )


@pytest.fixture(scope='module')
def c_layout() -> dict:
    result = subprocess.run(
        [str(_layout_binary())], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_every_dumped_struct_exists_in_the_python_mirror(c_layout):
    missing = [name for name in c_layout if not hasattr(_types, name)]
    assert not missing, f'structs in unirt.h with no ctypes mirror: {missing}'


def test_struct_sizes_match(c_layout):
    mismatched = {
        name: (info['size'], ctypes.sizeof(getattr(_types, name)))
        for name, info in c_layout.items()
        if ctypes.sizeof(getattr(_types, name)) != info['size']
    }
    assert not mismatched, f'C size vs ctypes size: {mismatched}'


def test_field_offsets_match(c_layout):
    """The failure this exists for: same size, different offsets.

    Appending a field can leave the total size unchanged while moving nothing
    -- but reordering, or adding into existing padding, shifts offsets with no
    size change at all, and every read past that point is wrong.
    """
    problems = []
    for name, info in c_layout.items():
        structure = getattr(_types, name)
        python_fields = {field[0]: getattr(structure, field[0]) for field in structure._fields_}
        for field_name, (offset, size) in info['fields'].items():
            descriptor = python_fields.get(field_name)
            if descriptor is None:
                problems.append(f'{name}.{field_name} is missing from the ctypes mirror')
                continue
            if descriptor.offset != offset:
                problems.append(
                    f'{name}.{field_name}: C offset {offset}, ctypes offset {descriptor.offset}'
                )
            if descriptor.size != size:
                problems.append(
                    f'{name}.{field_name}: C size {size}, ctypes size {descriptor.size}'
                )
    assert not problems, '\n'.join(problems)


def test_the_mirror_declares_no_extra_fields(c_layout):
    """A field only Python knows about is just as wrong: it makes the struct
    longer than the C side allocates, and the plugin writes short."""
    problems = []
    for name, info in c_layout.items():
        structure = getattr(_types, name)
        extra = {field[0] for field in structure._fields_} - set(info['fields'])
        if extra:
            problems.append(f'{name} has ctypes-only fields: {sorted(extra)}')
    assert not problems, '\n'.join(problems)

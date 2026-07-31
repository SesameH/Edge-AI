# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the release checks in tools/ci.

These run in CI on a wheel that is expensive to produce, so the parsing they
depend on is worth pinning here against real `auditwheel show` output.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / 'tools' / 'ci' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_manylinux_tag = _load('check_manylinux_tag')


# Trimmed from the manylinux_2_28_x86_64 job of run 30612375959: the wheel that
# first bundled ONNX Runtime. auditwheel wraps its prose, so the sentence this
# has to find is split across lines exactly as it is here.
REAL_REPORT = """
unirt-0.3.0-py3-none-manylinux_2_28_x86_64.whl is consistent with the
following platform tag: "linux_x86_64".

The wheel references external versioned symbols in these
system-provided shared libraries: libgcc_s.so.1 with versions
{'GCC_3.4', 'GCC_4.3.0'}, libc.so.6 with versions {'GLIBC_2.4',
'GLIBC_2.17'}, libonnxruntime.so.1 with versions {}

This constrains the platform tag to "manylinux_2_31_x86_64". In order
to achieve a more compatible tag, you would need to recompile a new
wheel from source on a system with earlier versions of these
libraries, such as a recent manylinux image.
"""

CLEAN_REPORT = """
unirt-0.3.0-py3-none-manylinux_2_31_x86_64.whl is consistent with the
following platform tag: "manylinux_2_17_x86_64".
"""


def test_finds_the_constraint_across_wrapped_lines():
    assert check_manylinux_tag.parse_required_tag(REAL_REPORT) == 'manylinux_2_31_x86_64'


def test_no_constraint_reported():
    assert check_manylinux_tag.parse_required_tag(CLEAN_REPORT) is None


def test_a_tag_that_undersells_the_glibc_floor_fails():
    """The regression that started this: ONNX Runtime needs 2.31, and the job
    was stamping 2.28 onto the wheel."""
    ok, message = check_manylinux_tag.check(REAL_REPORT, 'manylinux_2_28_x86_64')
    assert not ok
    assert '2.31' in message and '2_28' in message


def test_the_tag_we_now_ship_is_accepted():
    ok, message = check_manylinux_tag.check(REAL_REPORT, 'manylinux_2_31_x86_64')
    assert ok, message


def test_a_tag_stricter_than_required_is_accepted():
    ok, _ = check_manylinux_tag.check(REAL_REPORT, 'manylinux_2_34_x86_64')
    assert ok


def test_the_layout_verdict_alone_does_not_fail_the_build():
    """auditwheel calls the wheel 'linux_x86_64' because the SDK loads backends
    from its own plugin directories rather than <package>.libs. That is the
    design, and says nothing about glibc."""
    ok, _ = check_manylinux_tag.check(CLEAN_REPORT, 'manylinux_2_31_x86_64')
    assert ok


def test_a_non_manylinux_claim_is_rejected_when_a_floor_exists():
    ok, message = check_manylinux_tag.check(REAL_REPORT, 'linux_x86_64')
    assert not ok
    assert 'not a manylinux tag' in message


@pytest.mark.parametrize(('tag', 'expected'), [
    ('manylinux_2_31_x86_64', (2, 31)),
    ('manylinux_2_5_aarch64', (2, 5)),
    ('manylinux_2_28_aarch64', (2, 28)),
])
def test_glibc_floor_parsing(tag, expected):
    assert check_manylinux_tag.glibc_floor(tag) == expected

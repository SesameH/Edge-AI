# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that a Linux wheel's manylinux tag is not a lie about glibc.

The platform tag is stamped by hand at build time, so nothing else verifies the
binary earns it. A wheel that picked up a newer glibc than its tag promises
installs happily and then fails at import on the oldest distro the tag claims.

`auditwheel show` answers two different questions and only one of them is ours:

  "...is consistent with the following platform tag: linux_x86_64"

      Whether the wheel is a *self-contained* manylinux distribution by
      auditwheel's packaging rules. Ours is not, and cannot be: the SDK loads
      backends as plugins from unirt/lib/<plugin>/, resolved through the
      loader's own search, and auditwheel only recognises libraries vendored
      into <package>.libs with mangled sonames. It therefore reports every
      bundled backend -- libonnxruntime.so.1 today -- as an external
      dependency. That verdict is about layout, not portability.

  "This constrains the platform tag to manylinux_2_31_x86_64"

      The actual glibc floor, computed from the versioned symbols every
      library in the wheel references. That is what the tag must not undersell,
      and what this checks.

A bundled library that is genuinely absent still fails the build: it would be
missing from the wheel, and the smoke test that follows imports the package and
runs inference.
"""

from __future__ import annotations

import re
import subprocess
import sys

_CONSTRAINT = re.compile(r'constrains the platform tag to\s*"?(manylinux_[0-9]+_[0-9]+_\w+)"?')


def parse_required_tag(report: str) -> str | None:
    """Return the manylinux tag auditwheel says the wheel needs, if it says one.

    auditwheel wraps its prose at the terminal width, so the sentence arrives
    split across lines; joining first is what makes the match reliable.
    """

    return match.group(1) if (match := _CONSTRAINT.search(' '.join(report.split()))) else None


def glibc_floor(tag: str) -> tuple[int, int]:
    _, major, minor, _ = tag.split('_', 3)
    return int(major), int(minor)


def check(report: str, claimed_tag: str) -> tuple[bool, str]:
    required = parse_required_tag(report)
    if required is None:
        # No constraint reported means nothing in the wheel needs a newer glibc
        # than the build image provides.
        return True, f'auditwheel reports no glibc constraint; {claimed_tag} stands'
    if not claimed_tag.startswith('manylinux_'):
        return False, f'{claimed_tag} is not a manylinux tag, but the wheel needs {required}'
    need, claim = glibc_floor(required), glibc_floor(claimed_tag)
    if need > claim:
        return False, (
            f'wheel needs glibc {need[0]}.{need[1]} but is tagged {claimed_tag}, '
            f'which promises {claim[0]}.{claim[1]}'
        )
    return True, f'wheel needs glibc {need[0]}.{need[1]}, tag {claimed_tag} promises no more'


def main(wheel: str, claimed_tag: str) -> int:
    report = subprocess.run(
        ['auditwheel', 'show', wheel], capture_output=True, text=True, check=False
    )
    print(report.stdout)
    print(report.stderr, file=sys.stderr)
    ok, message = check(report.stdout, claimed_tag)
    print(message)
    if not ok:
        print(f'::error::{message}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

#!/bin/sh
# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause
#
# Populate installers/ and deps/win_{arm64,amd64} so the Windows guest can be
# verified with no internet access at all -- it only ever talks to this Mac.
# Neither directory is committed; both are reproducible from here.
#
# Run this on the Mac whenever pyproject.toml's dependencies change, or once on
# a fresh checkout.

set -eu

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)

# The pinned interpreter version and its expected hashes live in
# verify-wheels.ps1 (the guest checks them again before running an installer).
# Reading them back from there keeps one source of truth.
python_version=$(sed -n "s/^\$PythonVersion = '\(.*\)'/\1/p" "$here/verify-wheels.ps1")
[ -n "$python_version" ] || { echo 'cannot read $PythonVersion from verify-wheels.ps1' >&2; exit 1; }

mkdir -p "$here/installers"
for arch in arm64 amd64; do
    exe="$here/installers/python-$python_version-$arch.exe"
    if [ ! -f "$exe" ]; then
        echo "downloading python-$python_version-$arch.exe"
        curl -fL --retry 3 -o "$exe" \
            "https://www.python.org/ftp/python/$python_version/python-$python_version-$arch.exe"
    fi
    want=$(grep -A2 "Name = 'win_$arch'" "$here/verify-wheels.ps1" | sed -n "s/.*Sha256 = '\(.*\)'.*/\1/p")
    got=$(shasum -a 256 "$exe" | cut -d' ' -f1)
    [ "$want" = "$got" ] || { echo "$exe hashes $got, verify-wheels.ps1 expects $want" >&2; exit 1; }
    echo "installers/python-$python_version-$arch.exe ok"
done

# Read the requirements from pyproject.toml rather than repeating them here:
# a stale copy of this list is exactly how a dependency goes missing.
reqs=$(python3 - "$repo/bindings/python/pyproject.toml" <<'PY'
import sys, tomllib
with open(sys.argv[1], 'rb') as fh:
    print('\n'.join(tomllib.load(fh)['project']['dependencies']))
PY
)
[ -n "$reqs" ] || { echo 'no dependencies found in pyproject.toml' >&2; exit 1; }

# Cross-platform `pip download` resolves environment markers against *this*
# machine, so `colorama; platform_system == "Windows"` (pulled in by tqdm and
# click) silently drops out and only surfaces as a resolution failure inside
# the guest. Anything behind a Windows-only marker has to be named here; the
# closure check at the bottom is what catches an addition to this list.
windows_only='colorama'

for arch in arm64 amd64; do
    out="$here/deps/win_$arch"
    rm -rf "$out"
    # NUL-separated: a requirement may legally contain spaces ("tqdm >= 4.66").
    printf '%s\n' "$reqs" | tr '\n' '\0' | xargs -0 python3 -m pip download \
        --only-binary=:all: \
        --platform "win_$arch" --python-version 3.12 --implementation cp \
        -d "$out"
    # --no-deps: these are leaves, and their own markers are already resolved.
    python3 -m pip download --only-binary=:all: --no-deps \
        --platform "win_$arch" --python-version 3.12 --implementation cp \
        -d "$out" $windows_only
    echo "win_$arch: $(ls -1 "$out" | wc -l | tr -d ' ') wheels"
done

# Cross-check: with markers evaluated for the *guest* (Windows, CPython 3.12),
# nothing in the closure may require a package that is not staged. This is the
# check that would have caught colorama before it failed inside the VM.
python3 - "$here/deps" <<'PY'
import pathlib, re, sys, zipfile
from packaging.requirements import Requirement

ENV = {
    'os_name': 'nt', 'sys_platform': 'win32', 'platform_system': 'Windows',
    'platform_python_implementation': 'CPython',
    'python_version': '3.12', 'python_full_version': '3.12.8',
    'implementation_name': 'cpython', 'extra': '',
}

root = pathlib.Path(sys.argv[1])
failed = False
for d in sorted(root.iterdir()):
    env = dict(ENV, platform_machine='ARM64' if 'arm64' in d.name else 'AMD64')
    have = {re.split(r'-\d', w.name)[0].replace('_', '-').lower() for w in d.glob('*.whl')}
    for w in sorted(d.glob('*.whl')):
        z = zipfile.ZipFile(w)
        meta = next(n for n in z.namelist() if n.endswith('METADATA'))
        for line in z.read(meta).decode('utf-8', 'replace').splitlines():
            if not line.startswith('Requires-Dist:'):
                continue
            req = Requirement(line.split(':', 1)[1].strip())
            # No extras are requested, so an extra-guarded requirement is not
            # installed; `extra: ''` in the environment makes them evaluate false.
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            name = req.name.replace('_', '-').lower()
            if name not in have:
                print(f'{d.name}: {w.name} needs {name}, not staged')
                failed = True
if failed:
    raise SystemExit(1)
print('dependency closure complete for both architectures')
PY

#!/bin/sh
# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause
#
# Stage everything the Windows VM needs and serve it over HTTP.
#
# UTM's default shared networking puts the guest on a NAT behind the Mac, so
# the guest can reach the Mac on its LAN address. Shared folders would need
# SPICE guest tools installed first, which is one more thing to get wrong
# before any verification happens.
#
# Usage:  ./serve-kit.sh <wheel-dir> [port]
#
#   <wheel-dir>  directory holding this release's wheels, e.g.
#                .release-work-v0.2.2/final
#   [port]       default 8000

set -eu

if [ $# -lt 1 ]; then
    echo "usage: $0 <wheel-dir> [port]" >&2
    exit 2
fi

wheel_dir=$(cd "$1" && pwd)
port="${2:-8000}"
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
stage="$here/stage"
model="$repo/models/SmolLM2-135M-Instruct-Q8_0.gguf"

arm64_wheel=$(ls "$wheel_dir"/unirt-*-py3-none-win_arm64.whl)
amd64_wheel=$(ls "$wheel_dir"/unirt-*-py3-none-win_amd64.whl)
# unirt-0.2.2-py3-none-win_arm64.whl -> 0.2.2
version=$(basename "$arm64_wheel" | cut -d- -f2)

for f in "$amd64_wheel" "$model"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

if [ ! -d "$here/deps/win_arm64" ] || [ ! -d "$here/installers" ]; then
    echo "deps/ or installers/ missing -- run ./fetch-deps.sh first" >&2
    exit 1
fi

rm -rf "$stage"
mkdir -p "$stage"

# Keep the real filenames: pip rejects anything that is not a PEP 427 wheel
# name ("is not a valid wheel filename"), and the tag in the name is also what
# makes a mismatched install fail loudly.
cp "$arm64_wheel" "$amd64_wheel" "$stage/"
cp "$repo/bindings/python/smoke_inference_wheel.py" "$stage/smoke_inference_wheel.py"
cp "$model" "$stage/model.gguf"
cp "$here/verify-wheels.ps1" "$stage/verify-wheels.ps1"

# Both Python installers and the full dependency closure ride along so the
# guest never needs to reach the internet -- only the Mac. A fresh Windows
# guest has no network until the SPICE drivers are in, and that is exactly when
# you want to be running this.
cp "$here"/installers/python-*.exe "$stage/"

for arch in win_arm64 win_amd64; do
    mkdir -p "$stage/deps/$arch"
    cp "$here/deps/$arch"/*.whl "$stage/deps/$arch/"
    # An explicit manifest, rather than letting pip's --find-links scrape
    # http.server's directory listing: the listing format is not a contract.
    (cd "$stage/deps/$arch" && ls -1 ./*.whl | sed 's|^\./||' > MANIFEST.txt)
done

ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo '<mac-ip>')

echo "staged $version:"
ls -1sh "$stage"
echo
echo "In the Windows VM, run:"
echo
# printf, not echo: sh's echo expands backslash escapes, and `\v` in the
# PowerShell path turns into a vertical tab.
printf '  cd $env:USERPROFILE\n'
printf '  iwr http://%s:%s/verify-wheels.ps1 -OutFile verify-wheels.ps1\n' "$ip" "$port"
printf '  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force\n'
printf '  .\\verify-wheels.ps1 -KitUrl http://%s:%s -WheelVersion %s\n' "$ip" "$port" "$version"
echo
echo "serving $stage on port $port -- Ctrl-C to stop"
cd "$stage" && exec python3 -m http.server "$port"

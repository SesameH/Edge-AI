#!/usr/bin/env bash
# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause
#
# Put a Vulkan shader toolchain inside a manylinux_2_28 container.
#
# ggml-vulkan needs four things at build time: Vulkan headers, a Vulkan loader
# to link against, SPIRV-Headers, and glslc to compile its shaders. AlmaLinux 8
# packages the first two (1.3.283, new enough for the VK_EXT_layer_settings
# structs current ggml uses) and neither of the last two -- there is no shaderc
# in AppStream, EPEL or PowerTools for either architecture, and no prebuilt
# glslc that runs on glibc 2.28. So the shader half is built here from a pinned
# shaderc, which brings its own consistent glslang/SPIRV-Tools/SPIRV-Headers
# through git-sync-deps; installing that same SPIRV-Headers is what keeps
# ggml's find_package() agreeing with the compiler that produced its SPIR-V.
#
# Roughly ten minutes cold, which is why the caller caches the prefix. Nothing
# here is architecture-specific: x86_64 and aarch64 take the identical path.
#
#     tools/ci/setup_vulkan_manylinux.sh <prefix>
#
# Afterwards, configure with:
#     -DCMAKE_PREFIX_PATH=<prefix> -DVulkan_GLSLC_EXECUTABLE=<prefix>/bin/glslc

set -euo pipefail

SHADERC_TAG=v2026.3

prefix=${1:?usage: setup_vulkan_manylinux.sh <prefix>}
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)

# Unconditional: these land in /usr, which no cache of $prefix restores. The
# cache below skips only what is built into $prefix.
echo "::group::Vulkan headers and loader from AppStream"
dnf -y install vulkan-headers vulkan-loader-devel
echo "::endgroup::"

# Both halves have to be present to call it a hit: a prefix holding one of them
# is a half-written cache entry, and the build failure that causes points at
# CMake rather than at here.
if [ -x "$prefix/bin/glslc" ] && [ -d "$prefix/include/spirv" ]; then
    echo "shader toolchain already built in $prefix"
    "$prefix/bin/glslc" --version | head -1
    exit 0
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

echo "::group::Fetch shaderc $SHADERC_TAG and its pinned dependencies"
git -C "$work" clone --depth 1 --branch "$SHADERC_TAG" \
    https://github.com/google/shaderc.git shaderc
# Pinned in shaderc's own DEPS file, so the versions are ones it is tested with.
python3 "$work/shaderc/utils/git-sync-deps"
echo "::endgroup::"

echo "::group::Install SPIRV-Headers"
cmake -S "$work/shaderc/third_party/spirv-headers" -B "$work/build-spirv-headers" \
    -G Ninja -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --install "$work/build-spirv-headers"
echo "::endgroup::"

echo "::group::Build glslc"
# Only the glslc target: shaderc's libraries, tests and its other tools are
# nothing this build links or runs.
cmake -S "$work/shaderc" -B "$work/build-shaderc" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DSHADERC_SKIP_TESTS=ON \
    -DSHADERC_SKIP_EXAMPLES=ON \
    -DSHADERC_SKIP_COPYRIGHT_CHECK=ON \
    -DSPIRV_SKIP_EXECUTABLES=ON \
    -DENABLE_GLSLANG_BINARIES=OFF
cmake --build "$work/build-shaderc" --target glslc_exe -j"$(nproc)"
install -D -m755 "$work/build-shaderc/glslc/glslc" "$prefix/bin/glslc"
echo "::endgroup::"

"$prefix/bin/glslc" --version | head -1

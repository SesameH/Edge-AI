# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause
#
# Merges libunirt + the llama_cpp plugin + llama.cpp's own libggml*/libllama/
# libmtmd into ONE dynamic library, for wrapping into an XCFramework (see
# ../README.md's "Build the XCFramework" section) — the iOS equivalent of
# the Android AAR's auto-bundled jniLibs, so a consuming app adds one SPM
# binaryTarget dependency instead of manually linking five archives and
# embedding five loose dylibs.
#
# The sdk/plugins/llama_cpp static build (UNIRT_BUILD_SHARED=OFF) already
# produces every piece as a plain archive; the only thing this project adds
# is one dylib target that -force_load's libunirt.a and libunirt_llama_cpp.a
# so every C API entry point and the plugin's unirt_plugin_id/unirt_plugin_open
# symbols survive (nothing here references them directly, so a normal link
# would drop whatever isn't transitively reachable). llama.cpp's own static
# libs don't need force_load: the plugin genuinely calls deep into their API,
# so ordinary linking already retains what's used.
#
#   cmake -S bindings/ios/framework -B build-ios-framework \
#     -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator \
#     -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
#     -DGGML_METAL=OFF -DUNIRT_PLUGIN_MLX=OFF -DUNIRT_PLUGIN_ONNXRUNTIME=OFF
#   cmake --build build-ios-framework -j8
#
# Output: libunirt_ios.dylib (one file, no sibling dylibs needed).

cmake_minimum_required(VERSION 3.16)
project(unirt-ios-framework CXX)

get_filename_component(UNIRT_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

set(UNIRT_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(UNIRT_NATIVE_TESTS OFF CACHE BOOL "" FORCE)
set(UNIRT_PYTHON_TESTS OFF CACHE BOOL "" FORCE)
set(UNIRT_BENCHMARK OFF CACHE BOOL "" FORCE)

add_subdirectory("${UNIRT_REPO_ROOT}/sdk" sdk)

add_library(unirt_ios SHARED empty.cpp)
target_include_directories(unirt_ios PRIVATE "${UNIRT_REPO_ROOT}/sdk/include")
target_link_libraries(unirt_ios PRIVATE unirt unirt_llama_cpp)
target_link_options(unirt_ios PRIVATE
    -Wl,-force_load,$<TARGET_FILE:unirt>
    -Wl,-force_load,$<TARGET_FILE:unirt_llama_cpp>
    -install_name @rpath/libunirt_ios.dylib
)
set_target_properties(unirt_ios PROPERTIES OUTPUT_NAME "unirt_ios")

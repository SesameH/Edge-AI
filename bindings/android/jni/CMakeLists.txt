# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause
#
# Builds libunirt + the llama_cpp plugin + the JNI glue for Android.
#
#   cmake -S bindings/android/jni -B build-android \
#     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DUNIRT_PLUGIN_MLX=OFF -DUNIRT_PLUGIN_ONNXRUNTIME=OFF
#   cmake --build build-android -j
#
# Outputs: libunirt.so, libunirt_plugin_llama_cpp.so (flat naming for app
# lib dirs), libunirt_jni.so. Ship all three in jniLibs/arm64-v8a.

cmake_minimum_required(VERSION 3.22)
project(unirt-android CXX)

get_filename_component(UNIRT_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

add_subdirectory("${UNIRT_REPO_ROOT}/sdk" sdk)

add_library(unirt_jni SHARED unirt_jni.cpp)
target_include_directories(unirt_jni PRIVATE "${UNIRT_REPO_ROOT}/sdk/include")
target_link_libraries(unirt_jni PRIVATE unirt)
target_compile_definitions(unirt_jni PRIVATE UNIRT_SHARED)

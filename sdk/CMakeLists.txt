cmake_minimum_required(VERSION 3.16)
# C compiler is required by native dependencies.
project(unirt C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set RPATH configuration for all targets to use relative paths.
# macOS spells the loader-relative token @loader_path, not $ORIGIN, and we
# leave BUILD_WITH_INSTALL_RPATH off there so the build tree keeps CMake's
# automatic absolute rpaths (plugins live in a different dir than the
# llama.cpp dylibs they link).
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/..")
else()
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()

# Set UTF-8 encoding for MSVC
add_compile_options("$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/utf-8>")
add_compile_options("$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/utf-8>")

if(WIN32)
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

set(UNIRT_VERSION "v0.0.0" CACHE STRING "UniRT library version")

# ===============================================================================
# Build Options
# ===============================================================================

option(UNIRT_BUILD_SHARED      "Build libunirt and plugins as shared libraries" ON)
option(UNIRT_PLUGIN_LLAMA_CPP  "Enable llama.cpp backend"       ON)
option(UNIRT_PLUGIN_MLX        "Enable MLX backend (Apple)"     ON)
option(UNIRT_PLUGIN_ONNXRUNTIME "Enable ONNX Runtime embedding backend" ON)
# Vulkan reaches NVIDIA, AMD, Intel and the mobile vendors from a single build,
# with no vendor SDK on the build machine and no per-vendor wheel. Off by
# default because it needs Vulkan headers, a loader and glslc present; the
# Linux release job turns it on. See plugins/llama_cpp/CMakeLists.txt for why
# it also implies GGML_BACKEND_DL.
option(UNIRT_LLAMA_VULKAN      "Build the llama.cpp backend with Vulkan" OFF)
# A shipped binary cannot be built for the machine that built it, so the CPU
# backend is otherwise the target's baseline -- SSE2 on x86-64, with no AVX2.
# This builds one CPU backend per instruction-set level and lets ggml pick the
# best the running machine supports. Off by default because it multiplies both
# build time and installed size; the release jobs turn it on. See
# plugins/llama_cpp/CMakeLists.txt.
option(UNIRT_LLAMA_CPU_VARIANTS "Build one llama.cpp CPU backend per instruction-set level" OFF)
option(UNIRT_BENCHMARK         "Build benchmark example"        OFF)
option(UNIRT_PYTHON_TESTS      "Register Python architecture tests with CTest" ON)
option(UNIRT_NATIVE_TESTS      "Build native validation tests"   ON)

# Upstream MLX defaults an unspecified deployment target to the current host
# macOS version. Mirror that default for the plugin wrapper so its final link
# never combines a newer libmlx.a with an older UniRT deployment target. Users
# targeting an older supported macOS should pass the same explicit
# CMAKE_OSX_DEPLOYMENT_TARGET to both MLX and UniRT builds.
if(APPLE AND UNIRT_PLUGIN_MLX AND CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
   CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    execute_process(
        COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE UNIRT_HOST_MACOS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${UNIRT_HOST_MACOS_VERSION}")
    message(STATUS "UniRT deployment target follows MLX default: macOS ${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

message(STATUS "=============== Options ===============")
message(STATUS "UNIRT_VERSION:          ${UNIRT_VERSION}")
message(STATUS "UNIRT_PLUGIN_LLAMA_CPP: ${UNIRT_PLUGIN_LLAMA_CPP}")
message(STATUS "UNIRT_PLUGIN_MLX:       ${UNIRT_PLUGIN_MLX}")
message(STATUS "UNIRT_PLUGIN_ONNXRUNTIME: ${UNIRT_PLUGIN_ONNXRUNTIME}")
message(STATUS "UNIRT_LLAMA_VULKAN:     ${UNIRT_LLAMA_VULKAN}")
message(STATUS "UNIRT_LLAMA_CPU_VARIANTS: ${UNIRT_LLAMA_CPU_VARIANTS}")
message(STATUS "UNIRT_BENCHMARK:        ${UNIRT_BENCHMARK}")
message(STATUS "=======================================")

enable_testing()

# headers

set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# compile definitions

add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# subdirectories

add_subdirectory(./src)
add_subdirectory(./plugins)
if(UNIRT_BENCHMARK)
    add_subdirectory(./benchmark)
endif()

if(UNIRT_NATIVE_TESTS AND UNIRT_PLUGIN_MLX)
    add_executable(
        unirt-mlx-tokenizer-test
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/native/test_mlx_tokenizer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/plugins/mlx/src/tokenizer.cpp")
    target_include_directories(
        unirt-mlx-tokenizer-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/plugins/mlx/src")
    add_test(
        NAME unirt-mlx-tokenizer
        COMMAND unirt-mlx-tokenizer-test
                "${CMAKE_CURRENT_SOURCE_DIR}/../models/SmolLM2-135M-Instruct/tokenizer.json")
    set_tests_properties(unirt-mlx-tokenizer PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(UNIRT_NATIVE_TESTS)
    # Prints the C layout of every struct the Python binding mirrors, for
    # tests/python/test_abi_layout.py to compare against ctypes. Header-only:
    # it links nothing, so it builds in every configuration.
    add_executable(
        unirt-abi-layout-test
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/native/test_abi_layout.cpp")
    target_include_directories(unirt-abi-layout-test PRIVATE ${COMMON_INCLUDE})
    set_target_properties(unirt-abi-layout-test PROPERTIES MACOSX_BUNDLE FALSE)
    add_test(NAME unirt-abi-layout COMMAND unirt-abi-layout-test)

    add_executable(
        unirt-static-registration-test
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/native/test_static_registration.cpp")
    target_include_directories(unirt-static-registration-test PRIVATE ${COMMON_INCLUDE})
    target_link_libraries(unirt-static-registration-test PRIVATE unirt)
    # Linux builds run with INSTALL_RPATH ($ORIGIN) even in the build tree,
    # and the test binaries live one directory above libunirt.so.
    set_target_properties(unirt-static-registration-test PROPERTIES
        INSTALL_RPATH "$ORIGIN/src;$ORIGIN"
        MACOSX_BUNDLE FALSE)  # iOS defaults executables to .app bundles
    if(UNIRT_BUILD_SHARED)
        target_compile_definitions(unirt-static-registration-test PRIVATE UNIRT_SHARED)
    endif()
    add_test(NAME unirt-static-registration COMMAND unirt-static-registration-test)
endif()

if(UNIRT_NATIVE_TESTS AND UNIRT_PLUGIN_LLAMA_CPP)
    add_executable(
        unirt-inference-smoke-test
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/native/test_inference_smoke.cpp")
    target_include_directories(unirt-inference-smoke-test PRIVATE ${COMMON_INCLUDE})
    target_link_libraries(unirt-inference-smoke-test PRIVATE unirt)
    set_target_properties(unirt-inference-smoke-test PROPERTIES
        INSTALL_RPATH "$ORIGIN/src;$ORIGIN"
        MACOSX_BUNDLE FALSE)
    if(UNIRT_BUILD_SHARED)
        target_compile_definitions(unirt-inference-smoke-test PRIVATE UNIRT_SHARED)
    else()
        # Static build: link the backend in and register it in-process.
        target_compile_definitions(unirt-inference-smoke-test PRIVATE UNIRT_SMOKE_STATIC_PLUGIN)
        target_link_libraries(unirt-inference-smoke-test PRIVATE unirt_llama_cpp)
    endif()
    add_test(
        NAME unirt-inference-smoke
        COMMAND unirt-inference-smoke-test
                "${CMAKE_CURRENT_SOURCE_DIR}/../models/SmolLM2-135M-Instruct-Q8_0.gguf")
    # Plugins come from the installed tree (run `cmake --install` first —
    # already the standard dev loop): the build tree's plugin cannot resolve
    # its llama.cpp dependencies under Linux's $ORIGIN-only runpaths.
    set_tests_properties(unirt-inference-smoke PROPERTIES
        SKIP_RETURN_CODE 77
        ENVIRONMENT "UNIRT_PLUGIN_PATH=${CMAKE_SOURCE_DIR}/pkg-unirt/lib")
endif()

if(UNIRT_PYTHON_TESTS)
    set(UNIRT_PYTHON_TEST_EXECUTABLE "" CACHE FILEPATH "Python interpreter used for UniRT tests")
    if(NOT UNIRT_PYTHON_TEST_EXECUTABLE)
        # Prefer the user's PATH over a Python discovered indirectly through
        # CMAKE_PREFIX_PATH; the latter often lacks the binding test deps.
        find_program(
            UNIRT_PYTHON_FROM_PATH
            NAMES python3 python
            NO_CMAKE_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH)
        if(UNIRT_PYTHON_FROM_PATH)
            set(
                UNIRT_PYTHON_TEST_EXECUTABLE "${UNIRT_PYTHON_FROM_PATH}"
                CACHE FILEPATH "Python interpreter used for UniRT tests" FORCE)
        endif()
    endif()
    if(UNIRT_PYTHON_TEST_EXECUTABLE)
        execute_process(
            COMMAND "${UNIRT_PYTHON_TEST_EXECUTABLE}" -c "import pytest, huggingface_hub, tokenizers"
            RESULT_VARIABLE UNIRT_PYTHON_TEST_DEPS_RESULT
            OUTPUT_QUIET
            ERROR_QUIET)
        if(UNIRT_PYTHON_TEST_DEPS_RESULT EQUAL 0)
            get_filename_component(UNIRT_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
            add_test(
                NAME unirt-python-architecture
                COMMAND "${UNIRT_PYTHON_TEST_EXECUTABLE}" -m pytest tests/python -q)
            set_tests_properties(unirt-python-architecture PROPERTIES
                WORKING_DIRECTORY "${UNIRT_REPOSITORY_ROOT}"
                ENVIRONMENT
                    "PYTHONPATH=${UNIRT_REPOSITORY_ROOT}/bindings/python;UNIRT_LIB_PATH=$<TARGET_FILE:unirt>;UNIRT_PLUGIN_PATH=${CMAKE_BINARY_DIR}/plugins")
        else()
            message(STATUS "Python architecture test not registered: pytest/huggingface_hub/tokenizers unavailable")
        endif()
    endif()
endif()

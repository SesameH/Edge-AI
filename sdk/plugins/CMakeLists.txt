if(UNIRT_PLUGIN_LLAMA_CPP)
    add_subdirectory(./llama_cpp)
endif()

if(UNIRT_PLUGIN_MLX)
    add_subdirectory(./mlx)
endif()

if(UNIRT_PLUGIN_ONNXRUNTIME)
    add_subdirectory(./onnxruntime)
endif()

if(WIN32 AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    find_file(_openmp_dll
        NAMES libomp140.aarch64.dll libomp.dll
        HINTS
            "$ENV{VCToolsRedistDir}/arm64/Microsoft.VC143.OpenMP.LLVM"
            "$ENV{VCINSTALLDIR}/Tools/Llvm/ARM64/bin"
            "$ENV{VCINSTALLDIR}/Tools/Llvm/bin"
    )
    if(NOT _openmp_dll)
        message(FATAL_ERROR "OpenMP runtime DLL not found")
    endif()

    install(FILES "${_openmp_dll}"
            DESTINATION lib/llama_cpp
            RENAME "libomp140.aarch64.dll")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Some embedded ARM64 Linux images do not ship libgomp; bundle it so the wheel is self-contained.
    execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libgomp.so.1
        OUTPUT_VARIABLE _libgomp_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0 OR _libgomp_path STREQUAL "libgomp.so.1" OR NOT EXISTS "${_libgomp_path}")
        message(FATAL_ERROR "Cross toolchain cannot locate libgomp.so.1 (got '${_libgomp_path}').")
    endif()
    # Resolve symlink so install(FILES) copies the blob, not a dangling link.
    file(REAL_PATH "${_libgomp_path}" _libgomp_real)
    install(FILES "${_libgomp_real}"
            DESTINATION lib/llama_cpp
            RENAME "libgomp.so.1")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # ANDROID_NDK is a CMake variable the NDK's own toolchain file always
    # populates (from -DANDROID_NDK=... or auto-detected from the toolchain
    # file's location) regardless of who drives the build. $ENV{ANDROID_NDK_ROOT}
    # only happens to be set in our own CI/README invocations — Android
    # Studio's Gradle plugin (externalNativeBuild) never sets it, and would
    # otherwise fail this check on every AGP-driven build.
    file(GLOB_RECURSE _android_libomp
        "${ANDROID_NDK}/toolchains/llvm/prebuilt/*/libomp.so")
    list(FILTER _android_libomp INCLUDE REGEX "/lib/linux/aarch64/libomp.so$")
    if(NOT _android_libomp)
        message(FATAL_ERROR "Android NDK libomp.so not found under ${ANDROID_NDK}")
    endif()
    list(GET _android_libomp 0 _android_libomp)
    install(FILES "${_android_libomp}" DESTINATION lib/llama_cpp)
endif()

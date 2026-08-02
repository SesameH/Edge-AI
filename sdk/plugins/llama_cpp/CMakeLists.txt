# llama.cpp backend plugin.

set(GGML_BLAS OFF)
# GGML_NATIVE bakes the *build* machine's instruction set into the binary,
# which is right for a local build and wrong for anything shipped.
set(GGML_NATIVE OFF)

# Both features below need every ggml backend built as its own loadable module
# rather than linked into libggml. Either one turning it on is enough.
set(_unirt_backend_dl OFF)

if(UNIRT_LLAMA_CPU_VARIANTS)
    if(NOT UNIRT_BUILD_SHARED)
        message(FATAL_ERROR "UNIRT_LLAMA_CPU_VARIANTS requires UNIRT_BUILD_SHARED")
    endif()
    # GGML_NATIVE off means the CPU backend is built for the baseline of the
    # target architecture -- on x86-64 that is SSE2, with no AVX2 and no FMA,
    # which is what a portable wheel has to promise and what every machine
    # running it then gets. GGML_CPU_ALL_VARIANTS instead builds one CPU
    # backend per instruction-set level (x64, sse42, haswell, skylakex, zen4,
    # ...) and ggml_backend_load_best() picks the highest-scoring one the
    # machine actually supports, at load time. Portable wheel, native speed.
    #
    # ggml has no variant set for every target; the one we ship into is ARM on
    # Windows, where it would fail deep inside the subtree instead of here.
    if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^([Aa][Rr][Mm]64|aarch64)$")
        message(FATAL_ERROR
            "UNIRT_LLAMA_CPU_VARIANTS: ggml defines no CPU variant set for ARM on Windows")
    endif()
    # ggml's ARM-on-Linux set reaches armv9.2+sme, which GCC only understands
    # from 14 (Clang from 18). An older toolchain gets no diagnostic until it
    # is a hundred translation units into the build and cc1 rejects the
    # -march string, which reads as a broken checkout rather than a compiler
    # that is too old. manylinux_2_28 ships GCC 14; Ubuntu 24.04's default
    # GCC 13 does not.
    if(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-march=armv9.2-a+sme" UNIRT_CC_ACCEPTS_ARM_SME)
        if(NOT UNIRT_CC_ACCEPTS_ARM_SME)
            message(FATAL_ERROR
                "UNIRT_LLAMA_CPU_VARIANTS: ${CMAKE_C_COMPILER_ID} "
                "${CMAKE_C_COMPILER_VERSION} rejects -march=armv9.2-a+sme, which ggml's "
                "ARM variant set needs. Use GCC 14+/Clang 18+, or leave the option off.")
        endif()
    endif()
    set(GGML_CPU_ALL_VARIANTS ON)
    set(_unirt_backend_dl ON)
endif()

if(UNIRT_LLAMA_VULKAN)
    if(NOT UNIRT_BUILD_SHARED)
        # GGML_BACKEND_DL, which the next comment explains why Vulkan requires,
        # has nothing to dlopen in a static build.
        message(FATAL_ERROR "UNIRT_LLAMA_VULKAN requires UNIRT_BUILD_SHARED")
    endif()
    set(GGML_VULKAN ON)
    # ggml-vulkan links the Vulkan loader (Vulkan::Vulkan), so with the backends
    # compiled into libggml the whole plugin acquires a hard dependency on
    # libvulkan.so.1 / vulkan-1.dll. A machine with no Vulkan driver -- a slim
    # container, a headless server, a fresh Windows VM -- would then fail to
    # load the plugin at all and lose CPU inference with it: the same shape as
    # the OpenSSL dependency the LLAMA_OPENSSL block below removes.
    #
    # GGML_BACKEND_DL builds each ggml backend as its own loadable module and
    # skips the ones that will not load, so an absent Vulkan loader costs
    # exactly the Vulkan backend and nothing else. It also makes the CPU
    # backend a module, so the install and the wheel must carry it; the smoke
    # tests decode a real prompt, which fails loudly if they do not.
    set(_unirt_backend_dl ON)

    # ggml-vulkan.cpp includes <spirv/unified1/spirv.hpp>, but ggml only
    # find_package()s SPIRV-Headers and never links the imported target it
    # defines. So the include works exactly where those headers already sit on
    # the default search path -- Ubuntu's spirv-headers package does, and a
    # SPIRV-Headers installed into any other prefix (a Vulkan SDK, or the one
    # tools/ci/setup_vulkan_manylinux.sh builds) does not, failing at the first
    # compile rather than at configure. Put the found headers on the path for
    # the llama.cpp subtree, which is where they are needed.
    find_package(SPIRV-Headers CONFIG QUIET)
    if(TARGET SPIRV-Headers::SPIRV-Headers)
        get_target_property(_unirt_spirv_includes
            SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
        if(_unirt_spirv_includes)
            include_directories(${_unirt_spirv_includes})
        endif()
    endif()
endif()

set(GGML_BACKEND_DL ${_unirt_backend_dl})
if(_unirt_backend_dl)
    # The backend modules are the only thing this build installs to a bindir,
    # and ggml_add_backend_library() sends them there. Point that at the
    # plugin's own directory rather than accept a second copy under bin/.
    set(CMAKE_INSTALL_BINDIR "lib/llama_cpp")
endif()

# common is needed solely for json_schema_to_grammar() (JSON Schema →
# GBNF); everything else still goes through the public llama.h/mtmd API.
# CACHE FORCE: llama.cpp declares this with option(), which a plain set()
# in the parent scope does not override.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF)
# Both default ON in llama.cpp and both are for its own model downloader, which
# nothing here calls -- downloads go through huggingface_hub in the Python
# layer. Leaving OPENSSL on links libssl/libcrypto into llama-common whenever
# the build machine happens to have them, which is a runtime dependency the
# user's machine then has to satisfy: the published macOS wheel points at
# /opt/homebrew/opt/openssl@3, and on Windows the plugin fails to load outright
# with ERROR_MOD_NOT_FOUND, since the loader deliberately does not search PATH.
set(LLAMA_OPENSSL OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_MTMD ON)
set(MTMD_VIDEO OFF)
# Follow our own shared/static choice instead of always forcing llama.cpp's
# own ggml/llama/mtmd shared: a fully static UNIRT_BUILD_SHARED=OFF build
# (iOS today, and the merged-dylib build under bindings/ios/framework)
# needs these static too, or they'd still ship as separate dylibs the
# embedder has to find at runtime regardless of our own flag.
if(UNIRT_BUILD_SHARED)
    set(BUILD_SHARED_LIBS ON)
else()
    set(BUILD_SHARED_LIBS OFF)
endif()

add_subdirectory(
    ../../../third-party/llama.cpp
    ${CMAKE_BINARY_DIR}/third-party/llama.cpp
    EXCLUDE_FROM_ALL)

add_subdirectory(./src)

# Install the llama.cpp runtime dylibs next to the plugin so @loader_path /
# $ORIGIN resolution works in the installed layout.
# GGML_AVAILABLE_BACKENDS rather than a list of backend names written out
# here: with GGML_CPU_ALL_VARIANTS the CPU backend is not one target called
# ggml-cpu but a dozen (ggml-cpu-haswell, ggml-cpu-zen4, ...) whose names
# depend on the target architecture. ggml appends every backend it built to
# that cache variable, which is the only place the real list exists.
#
# It has to be installed from here at all because the subtree is added
# EXCLUDE_FROM_ALL, which drops its own install() rules -- so a backend
# missing from this loop is simply absent from the wheel, with the build
# still green.
set(LLAMA_LIBS mtmd llama llama-common ggml ggml-base ${GGML_AVAILABLE_BACKENDS})
foreach(dep IN LISTS LLAMA_LIBS)
    if(TARGET ${dep})
        # unversioned file names, so the plugin's install_name matches
        set_property(TARGET ${dep} PROPERTY VERSION)
        set_property(TARGET ${dep} PROPERTY SOVERSION)
        install(TARGETS ${dep}
            LIBRARY DESTINATION lib/llama_cpp
            RUNTIME DESTINATION lib/llama_cpp)
    endif()
endforeach()

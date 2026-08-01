# llama.cpp backend plugin.

set(GGML_BLAS OFF)
set(GGML_NATIVE OFF)

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
    set(GGML_BACKEND_DL ON)
    # These modules are the only thing this build installs to a bindir, and
    # ggml_add_backend_library() sends them there. Point that at the plugin's
    # own directory rather than accept a second copy under bin/.
    set(CMAKE_INSTALL_BINDIR "lib/llama_cpp")

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
else()
    set(GGML_BACKEND_DL OFF)
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
set(LLAMA_LIBS mtmd llama llama-common ggml ggml-base ggml-cpu ggml-metal ggml-blas ggml-vulkan ggml-cuda)
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

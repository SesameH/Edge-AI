# llama.cpp backend plugin.

set(GGML_BLAS OFF)
set(GGML_NATIVE OFF)
set(GGML_BACKEND_DL OFF)

set(LLAMA_BUILD_COMMON OFF)
set(LLAMA_CURL OFF)
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
set(LLAMA_LIBS mtmd llama ggml ggml-base ggml-cpu ggml-metal ggml-blas ggml-vulkan ggml-cuda)
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

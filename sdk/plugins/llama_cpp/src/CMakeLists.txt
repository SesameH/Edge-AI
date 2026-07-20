add_library(unirt_llama_cpp SHARED plugin.cpp llm.cpp vlm.cpp embedding.cpp)

target_compile_definitions(unirt_llama_cpp PUBLIC UNIRT_SHARED)

target_include_directories(unirt_llama_cpp PUBLIC ${COMMON_INCLUDE})

target_link_libraries(unirt_llama_cpp
    PUBLIC  llama mtmd
    PRIVATE unirt)

set_target_properties(unirt_llama_cpp PROPERTIES
    OUTPUT_NAME "unirt_plugin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/llama_cpp"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/llama_cpp"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/llama_cpp")

if(APPLE)
    set_target_properties(unirt_llama_cpp PROPERTIES
        INSTALL_RPATH "@loader_path;@loader_path/..")
elseif(UNIX)
    set_target_properties(unirt_llama_cpp PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
        BUILD_RPATH "$ORIGIN:$ORIGIN/..")
endif()

install(TARGETS unirt_llama_cpp
    LIBRARY DESTINATION lib/llama_cpp
    RUNTIME DESTINATION lib/llama_cpp)

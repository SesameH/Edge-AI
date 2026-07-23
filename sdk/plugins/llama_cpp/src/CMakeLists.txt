if(UNIRT_BUILD_SHARED)
    add_library(unirt_llama_cpp SHARED plugin.cpp llm.cpp vlm.cpp embedding.cpp schema_grammar.cpp)
    target_compile_definitions(unirt_llama_cpp PUBLIC UNIRT_SHARED)
else()
    # Static plugin: the app links it and calls
    # unirt_register_plugin(unirt_plugin_id, unirt_plugin_open).
    add_library(unirt_llama_cpp STATIC plugin.cpp llm.cpp vlm.cpp embedding.cpp schema_grammar.cpp)
endif()

target_include_directories(unirt_llama_cpp PUBLIC ${COMMON_INCLUDE})

target_link_libraries(unirt_llama_cpp
    PUBLIC  llama mtmd
    PRIVATE unirt llama-common)

if(NOT UNIRT_BUILD_SHARED)
    # Static archives keep the target name; discovery naming is irrelevant.
elseif(ANDROID)
    # Android packs every .so flat into the app's lib dir; the registry
    # discovers flat plugins by the libunirt_plugin_<id>.so naming.
    set_target_properties(unirt_llama_cpp PROPERTIES
        OUTPUT_NAME "unirt_plugin_llama_cpp"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins")
else()
    set_target_properties(unirt_llama_cpp PROPERTIES
        OUTPUT_NAME "unirt_plugin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/llama_cpp"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/llama_cpp"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/llama_cpp")
endif()

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

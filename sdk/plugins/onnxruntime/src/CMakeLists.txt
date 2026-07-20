add_library(unirt_onnxruntime SHARED plugin.cpp embedding.cpp)

target_compile_definitions(unirt_onnxruntime PUBLIC UNIRT_SHARED)
target_include_directories(
    unirt_onnxruntime PRIVATE
    ${COMMON_INCLUDE}
    ${UNIRT_ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(
    unirt_onnxruntime PRIVATE
    unirt
    ${UNIRT_ONNXRUNTIME_LIBRARY})

get_filename_component(_ort_library_dir "${UNIRT_ONNXRUNTIME_LIBRARY}" DIRECTORY)
set_target_properties(unirt_onnxruntime PROPERTIES
    CXX_STANDARD 17
    OUTPUT_NAME "unirt_plugin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/onnxruntime"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/onnxruntime"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/onnxruntime"
    BUILD_RPATH "${_ort_library_dir}")

if(APPLE)
    set_target_properties(unirt_onnxruntime PROPERTIES INSTALL_RPATH "@loader_path")
    install(FILES "${UNIRT_ONNXRUNTIME_RUNTIME}"
        DESTINATION lib/onnxruntime RENAME libonnxruntime.1.dylib)
elseif(UNIX)
    set_target_properties(unirt_onnxruntime PROPERTIES
        INSTALL_RPATH "$ORIGIN" BUILD_RPATH "${_ort_library_dir}")
    install(FILES "${UNIRT_ONNXRUNTIME_RUNTIME}"
        DESTINATION lib/onnxruntime RENAME libonnxruntime.so.1)
else()
    install(FILES "${UNIRT_ONNXRUNTIME_RUNTIME}" DESTINATION lib/onnxruntime)
endif()

install(TARGETS unirt_onnxruntime
    LIBRARY DESTINATION lib/onnxruntime
    RUNTIME DESTINATION lib/onnxruntime)

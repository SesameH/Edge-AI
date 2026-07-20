# ONNX Runtime embedding backend. The dependency stays external: point
# UNIRT_ONNXRUNTIME_ROOT at an extracted official SDK (include/ + lib/).

set(UNIRT_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime SDK prefix")

set(_ort_include_hints)
set(_ort_library_hints)
if(UNIRT_ONNXRUNTIME_ROOT)
    list(APPEND _ort_include_hints "${UNIRT_ONNXRUNTIME_ROOT}/include")
    list(APPEND _ort_library_hints "${UNIRT_ONNXRUNTIME_ROOT}/lib")
endif()

find_path(
    UNIRT_ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${_ort_include_hints})
find_library(
    UNIRT_ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${_ort_library_hints})

if(NOT UNIRT_ONNXRUNTIME_INCLUDE_DIR OR NOT UNIRT_ONNXRUNTIME_LIBRARY)
    message(WARNING
        "ONNX Runtime plugin skipped: set UNIRT_ONNXRUNTIME_ROOT to an SDK "
        "containing include/onnxruntime_cxx_api.h and lib/libonnxruntime")
    return()
endif()

if(WIN32)
    find_file(
        UNIRT_ONNXRUNTIME_RUNTIME
        NAMES onnxruntime.dll
        HINTS "${UNIRT_ONNXRUNTIME_ROOT}/lib" "${UNIRT_ONNXRUNTIME_ROOT}/bin")
    if(NOT UNIRT_ONNXRUNTIME_RUNTIME)
        message(FATAL_ERROR "ONNX Runtime import library found but onnxruntime.dll is missing")
    endif()
else()
    get_filename_component(
        UNIRT_ONNXRUNTIME_RUNTIME "${UNIRT_ONNXRUNTIME_LIBRARY}" REALPATH)
endif()

message(STATUS "ONNX Runtime include: ${UNIRT_ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${UNIRT_ONNXRUNTIME_RUNTIME}")

get_filename_component(_ort_sdk_prefix "${UNIRT_ONNXRUNTIME_INCLUDE_DIR}" DIRECTORY)
if(EXISTS "${_ort_sdk_prefix}/LICENSE")
    install(FILES "${_ort_sdk_prefix}/LICENSE"
        DESTINATION share/licenses/onnxruntime RENAME LICENSE)
endif()
if(EXISTS "${_ort_sdk_prefix}/ThirdPartyNotices.txt")
    install(FILES "${_ort_sdk_prefix}/ThirdPartyNotices.txt"
        DESTINATION share/licenses/onnxruntime)
endif()

add_subdirectory(src)

file(GLOB BRIDGE_SRC *.cpp)

add_library(unirt SHARED ${BRIDGE_SRC})

set_source_files_properties(version.cpp PROPERTIES
    COMPILE_DEFINITIONS UNIRT_VERSION="${UNIRT_VERSION}")
target_compile_definitions(unirt PRIVATE UNIRT_SHARED UNIRT_BUILD)

target_include_directories(unirt PUBLIC ${COMMON_INCLUDE})

install(TARGETS unirt
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib)

install(FILES ${CMAKE_SOURCE_DIR}/include/unirt.h DESTINATION include)

file(GLOB BRIDGE_SRC *.cpp)

if(UNIRT_BUILD_SHARED)
    add_library(unirt SHARED ${BRIDGE_SRC})
    target_compile_definitions(unirt PRIVATE UNIRT_SHARED UNIRT_BUILD)
else()
    # Static build (iOS forbids dlopen; embedders link one binary and use
    # unirt_register_plugin instead of the directory scan).
    add_library(unirt STATIC ${BRIDGE_SRC})
endif()

set_source_files_properties(version.cpp PROPERTIES
    COMPILE_DEFINITIONS UNIRT_VERSION="${UNIRT_VERSION}")

target_include_directories(unirt PUBLIC ${COMMON_INCLUDE})

install(TARGETS unirt
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib)

install(FILES ${CMAKE_SOURCE_DIR}/include/unirt.h DESTINATION include)

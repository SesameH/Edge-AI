add_library(unirt_mlx SHARED plugin.cpp model.cpp tokenizer.cpp json_constraint.cpp)

target_compile_definitions(unirt_mlx PUBLIC UNIRT_SHARED)

# MLX is built separately (see README): cmake -S third-party/mlx -B build-mlx ...
# and installed into build-mlx/install. Override with -DUNIRT_MLX_PREFIX.
set(UNIRT_MLX_PREFIX "${CMAKE_SOURCE_DIR}/../build-mlx/install" CACHE PATH "MLX install prefix")
find_package(MLX CONFIG REQUIRED PATHS "${UNIRT_MLX_PREFIX}" NO_DEFAULT_PATH)

target_include_directories(unirt_mlx PUBLIC ${COMMON_INCLUDE})

target_link_libraries(unirt_mlx PRIVATE unirt mlx)

set_target_properties(unirt_mlx PROPERTIES
    CXX_STANDARD 17
    OUTPUT_NAME "unirt_plugin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/mlx"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/mlx"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/mlx")

set_target_properties(unirt_mlx PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/..")

install(TARGETS unirt_mlx
    LIBRARY DESTINATION lib/mlx
    RUNTIME DESTINATION lib/mlx)

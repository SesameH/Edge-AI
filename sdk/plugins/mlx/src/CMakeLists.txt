add_library(unirt_mlx SHARED
    plugin.cpp model.cpp weight_cache.cpp tokenizer.cpp json_constraint.cpp)

target_compile_definitions(unirt_mlx PUBLIC UNIRT_SHARED)

# MLX is built separately (see README): cmake -S third-party/mlx -B build-mlx ...
# and installed into build-mlx/install. Override with -DUNIRT_MLX_PREFIX.
set(UNIRT_MLX_PREFIX "${CMAKE_SOURCE_DIR}/../build-mlx/install" CACHE PATH "MLX install prefix")

# Built against macOS SDK 26.2 or newer, MLX adds its JACCL backend (RDMA over
# Thunderbolt) and puts jaccl::jaccl in the exported link interface -- but it
# installs only jacclTargets.cmake, with no jacclConfig.cmake, so
# find_package(jaccl) cannot resolve it and find_package(MLX) then fails with
# "target was not found". Import the targets file directly when the install has
# one; an MLX built on an older SDK simply has no such file.
set(_unirt_jaccl_targets "${UNIRT_MLX_PREFIX}/lib/cmake/jaccl/jacclTargets.cmake")
if(EXISTS "${_unirt_jaccl_targets}")
    include("${_unirt_jaccl_targets}")
endif()

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

# MLX keeps its Metal kernels in a separate mlx.metallib, and the path compiled
# into the static library points at the machine that built MLX -- so a plugin
# shipped anywhere else finds nothing and reports no Metal device, with the
# backend otherwise loading fine. Its loader looks beside the calling binary
# first (load_colocated_library in mlx/backend/metal/device.cpp), so installing
# the metallib next to the plugin is what makes a relocatable build work.
#
# Build MLX with -DMLX_METAL_JIT=ON: that metallib is 1.2 MB against 124 MB for
# the fully precompiled one, which no wheel could carry. The kernels are then
# compiled on first use and cached by Metal.
set(_unirt_mlx_metallib "${UNIRT_MLX_PREFIX}/lib/mlx.metallib")
if(EXISTS "${_unirt_mlx_metallib}")
    install(FILES "${_unirt_mlx_metallib}" DESTINATION lib/mlx)
else()
    message(WARNING
        "MLX metallib not found at ${_unirt_mlx_metallib}; the installed "
        "plugin will fall back to the path baked in when MLX was built, which "
        "only exists on that machine")
endif()

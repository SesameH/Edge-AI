if(NOT APPLE)
    message(STATUS "MLX plugin skipped: Apple platforms only")
    return()
endif()

add_subdirectory(./src)

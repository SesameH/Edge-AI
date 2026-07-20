add_executable(unirt-bench benchmark.c)

target_include_directories(unirt-bench
    PRIVATE
        ${COMMON_INCLUDE}
)

target_link_libraries(unirt-bench
    PRIVATE unirt
)

# sqrt() in summarize_full() needs libm on glibc cross-compile targets
# (Linux aarch64). bionic / msvcrt fold sqrt into libc, but glibc keeps it
# in libm and the linker enforces --as-needed.
if(NOT WIN32 AND NOT ANDROID)
    target_link_libraries(unirt-bench PRIVATE m)
endif()

target_compile_definitions(unirt-bench PRIVATE UNIRT_SHARED)

set_target_properties(unirt-bench PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# The executable is installed in bin while libunirt is installed in lib.
# Override the library-oriented project default, which is correct for plugins
# but one directory too shallow for an executable.
if(APPLE)
    set_target_properties(unirt-bench PROPERTIES
        INSTALL_RPATH "@loader_path/../lib")
elseif(UNIX)
    set_target_properties(unirt-bench PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib")
endif()

install(TARGETS unirt-bench RUNTIME DESTINATION bin)

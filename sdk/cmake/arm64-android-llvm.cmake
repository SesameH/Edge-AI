set(ANDROID_NDK_ROOT "$ENV{ANDROID_NDK_ROOT}")
include("${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake")

set(CMAKE_C_FLAGS "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE")
set(CMAKE_CXX_FLAGS "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE")

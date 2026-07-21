// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// CMake's add_library() needs at least one source file; every symbol this
// merged dylib actually exports comes from -force_load'ing the static
// libunirt/libunirt_llama_cpp archives (see CMakeLists.txt), not from here.

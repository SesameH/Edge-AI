// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace unirt::build_config {

// Canonical plugin ids the SDK ships backends for.
inline constexpr char kPluginIdLlamaCpp[] = "llama_cpp";
inline constexpr char kPluginIdMlx[]      = "mlx";
inline constexpr char kPluginIdOnnxRuntime[] = "onnxruntime";

// Bridge version string, stamped by the build (see UNIRT_VERSION in CMake).
extern const char kBridgeVersion[];

}  // namespace unirt::build_config

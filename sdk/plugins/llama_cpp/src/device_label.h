// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

#include "ggml-backend.h"

namespace unirt::llama_plugin {

// ggml_backend_dev_description() alone is ambiguous on Apple Silicon: the
// Metal GPU device and the CPU device both describe themselves by the same
// chip name (e.g. "Apple M1"). Append the backend registry name so callers
// (chat.html's device card, runtime_stats() consumers) can actually tell
// which compute unit is in use.
inline std::string device_label(ggml_backend_dev_t device, const char* fallback) {
    if (!device) return fallback && fallback[0] ? fallback : "CPU";
    const char* description = ggml_backend_dev_description(device);
    std::string label = description && description[0]
                             ? description
                             : (fallback && fallback[0] ? fallback : "device");
    if (ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device)) {
        if (const char* reg_name = ggml_backend_reg_name(reg); reg_name && reg_name[0]) {
            label += " (";
            label += reg_name;
            label += ")";
        }
    }
    return label;
}

}  // namespace unirt::llama_plugin

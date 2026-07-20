// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// User-facing device-alias resolution: cpu / gpu / npu / hybrid / auto →
// a concrete device id plus an n_gpu_layers policy, per platform. Language
// bindings call this instead of duplicating the mapping.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

#include "unirt.h"
#include "logging.h"

namespace {

std::string canonical_alias(const char* raw) {
    if (!raw) return {};
    std::string value(raw);
    auto        head = value.find_first_not_of(" \t\r\n");
    if (head == std::string::npos) return {};
    auto tail = value.find_last_not_of(" \t\r\n");
    value     = value.substr(head, tail - head + 1);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

char* heap_string(const char* text) {
#if defined(_WIN32)
    return _strdup(text);
#else
    return strdup(text);
#endif
}

/** Resolution policy for one alias on this platform. */
struct AliasRule {
    const char*                alias;
    std::optional<const char*> device_id;   // nullopt = plugin default
    std::optional<int32_t>     ngl;         // nullopt = pass caller's value through
    const char*                coercion;    // non-null = emit this warning
};

// Aliases valid on every platform.
constexpr const char* kKnownAliases[] = {"cpu", "gpu", "npu", "hybrid"};

// Hardware-agnostic on every platform: aliases resolve to the backend's own
// default device (Metal on Apple, CUDA/Vulkan/CPU elsewhere — whatever the
// build ships), and "cpu" additionally pins n_gpu_layers to 0. NPU-class
// aliases degrade with a warning because no bundled backend drives an NPU;
// callers targeting specific silicon pass a concrete device id instead of
// an alias, which bypasses this resolver entirely.
constexpr AliasRule kRules[] = {
    {"cpu", std::nullopt, 0, nullptr},
    {"gpu", std::nullopt, std::nullopt, nullptr},
    {"npu", std::nullopt, std::nullopt,
     "no NPU backend in this build; using the backend's default device"},
    {"hybrid", std::nullopt, std::nullopt,
     "no hybrid scheduling in this build; using the backend's default device"},
};
constexpr const char* kDefaultAlias = "gpu";

}  // namespace

int32_t unirt_resolve_device(const unirt_ResolveDeviceInput* input, unirt_ResolveDeviceOutput* output) {
    if (output) *output = {};
    if (!input || !output) {
        UNIRT_LOG_ERROR("resolve_device: null input/output");
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    output->ngl = input->ngl_default;

    if (!input->plugin_id) {
        UNIRT_LOG_ERROR("resolve_device: plugin_id is null");
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }

    std::string alias = canonical_alias(input->mode);
    if (alias.empty() || alias == "auto") alias = kDefaultAlias;

    const bool known = std::any_of(
        std::begin(kKnownAliases), std::end(kKnownAliases),
        [&](const char* candidate) { return alias == candidate; });
    if (!known) {
        UNIRT_LOG_ERROR("resolve_device: unknown alias '{}'", alias);
        return UNIRT_ERROR_COMMON_INVALID_DEVICE;
    }

    for (const auto& rule : kRules) {
        if (alias != rule.alias) continue;
        if (rule.device_id) output->device_id = heap_string(*rule.device_id);
        if (rule.ngl) output->ngl = *rule.ngl;
        if (rule.coercion) output->warning = heap_string(rule.coercion);
        break;
    }
    return UNIRT_SUCCESS;
}

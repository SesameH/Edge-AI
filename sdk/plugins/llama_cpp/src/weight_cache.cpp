// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "weight_cache.h"

#include <filesystem>
#include <map>
#include <mutex>

#include "logging.h"

namespace unirt::llama_plugin {

namespace {

std::mutex& cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

// weak_ptr, not shared_ptr: the cache must not be what keeps weights alive, or
// closing every handle would still hold the model until the process exits.
// Entries whose weights are gone are pruned when they are next looked at.
std::map<std::string, std::weak_ptr<llama_model>>& cache() {
    static std::map<std::string, std::weak_ptr<llama_model>> entries;
    return entries;
}

std::string cache_key(
    const std::string& path, const std::string& device_key, const llama_model_params& params) {
    // Two spellings of one file are one model. Falls back to the path as
    // given when it cannot be resolved -- that load is about to fail anyway,
    // and llama.cpp's own error names the file the caller actually asked for.
    std::error_code error;
    const auto      canonical = std::filesystem::weakly_canonical(path, error);
    // \x1f between parts: a unit separator cannot appear in a path or a
    // device id, so no two different requests can collide on one key.
    return (error ? path : canonical.string()) + '\x1f' + device_key + '\x1f' +
           std::to_string(params.n_gpu_layers);
}

}  // namespace

SharedModel acquire_model(
    const std::string& path, const std::string& device_key, const llama_model_params& params) {
    const std::string key = cache_key(path, device_key, params);

    std::lock_guard<std::mutex> lock(cache_mutex());
    auto& entries = cache();

    if (auto found = entries.find(key); found != entries.end()) {
        if (SharedModel existing = found->second.lock()) {
            UNIRT_LOG_DEBUG("llama_cpp: reusing already-loaded weights for {}", path);
            return existing;
        }
        entries.erase(found);
    }

    llama_model* raw = llama_model_load_from_file(path.c_str(), params);
    if (!raw) return nullptr;

    SharedModel model(raw, [](llama_model* victim) {
        if (victim) llama_model_free(victim);
    });
    entries[key] = model;

    // Prune the keys whose weights have since been freed. Doing it here keeps
    // the map proportional to what is loaded rather than to how many distinct
    // models the process has ever opened.
    for (auto it = entries.begin(); it != entries.end();) {
        it = it->second.expired() ? entries.erase(it) : std::next(it);
    }
    return model;
}

}  // namespace unirt::llama_plugin

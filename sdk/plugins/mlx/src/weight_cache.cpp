// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// One resident copy of each checkpoint, shared by every backend instance that
// asked for it. Two decoding slots on the same model are two KV caches over
// one set of parameters, not two models -- the parameters are read-only once
// loaded, so there is nothing to keep apart.

#include <filesystem>
#include <map>
#include <mutex>
#include <string>

#include "logging.h"
#include "model.h"

namespace unirt::mlx_plugin {

namespace {

std::mutex& cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

// weak_ptr, not shared_ptr: the cache must not be what keeps weights alive, or
// closing every handle would still hold them until the process exits. Keys
// whose weights are gone are pruned on the next load.
std::map<std::string, std::weak_ptr<const LlamaWeights>>& cache() {
    static std::map<std::string, std::weak_ptr<const LlamaWeights>> entries;
    return entries;
}

// Two spellings of one directory are one checkpoint. Falls back to the string
// as given when the path cannot be resolved -- a path that does not exist is
// about to fail in load_weights() anyway, and reporting that error is better
// than reporting a canonicalization one.
std::string cache_key(const std::string& model_dir) {
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(model_dir, error);
    return error ? model_dir : canonical.string();
}

}  // namespace

SharedWeights acquire_weights(const std::string& model_dir, const LlamaConfig& cfg) {
    const std::string key = cache_key(model_dir);

    // The load happens under the lock, so two slots opening the same model at
    // once read it once rather than racing to read it twice. It also
    // serializes loads of *different* models, which only costs at startup.
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto&                       entries = cache();

    if (auto found = entries.find(key); found != entries.end()) {
        if (SharedWeights existing = found->second.lock()) {
            UNIRT_LOG_DEBUG("mlx: reusing already-loaded weights for {}", model_dir);
            return existing;
        }
        entries.erase(found);
    }

    auto weights = std::make_shared<const LlamaWeights>(load_weights(model_dir, cfg));
    entries[key] = weights;

    for (auto it = entries.begin(); it != entries.end();) {
        it = it->second.expired() ? entries.erase(it) : std::next(it);
    }
    return weights;
}

size_t loaded_weight_count() {
    std::lock_guard<std::mutex> lock(cache_mutex());
    size_t                      live = 0;
    for (const auto& [_, weak] : cache()) {
        if (!weak.expired()) ++live;
    }
    return live;
}

}  // namespace unirt::mlx_plugin

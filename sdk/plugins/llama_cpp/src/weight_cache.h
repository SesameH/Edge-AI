// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <string>

#include <llama.h>

namespace unirt::llama_plugin {

/** Weights held jointly by every handle that opened the same file. */
using SharedModel = std::shared_ptr<llama_model>;

/**
 * Load a GGUF's weights, or take a reference to the copy already in memory.
 *
 * A llama_model is read-only once loaded and llama.cpp supports any number of
 * contexts over one: it is the llama_context that holds the mutable KV state.
 * Loading per handle therefore bought nothing and cost a full extra copy of
 * the weights, which is what stopped a server from serving more than one
 * request at a time -- N slots meant N x the model in memory.
 *
 * Sharing is keyed on the file *and* on how it was asked for: a handle pinned
 * to a different device, or offloading a different number of layers, needs its
 * own load because those choices are baked into the loaded tensors.
 *
 * Freed when the last handle using it goes. Safe to call from several threads.
 */
SharedModel acquire_model(
    const std::string& path, const std::string& device_key, const llama_model_params& params);

/** How many distinct weight sets are currently loaded. For tests and logging. */
size_t loaded_model_count();

}  // namespace unirt::llama_plugin

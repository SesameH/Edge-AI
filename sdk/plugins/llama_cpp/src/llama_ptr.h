// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>

#include <llama.h>

namespace unirt::llama_plugin {

/** Sole ownership of a llama_model. Still what the VLM and embedding backends
 *  use: only the text backend has a reason to share weights so far, since only
 *  it gets opened several times over for a pool of decoding slots. */
struct ModelDeleter {
    void operator()(llama_model* model) const noexcept;
};

struct ContextDeleter {
    void operator()(llama_context* context) const noexcept;
};

struct SamplerDeleter {
    void operator()(llama_sampler* sampler) const noexcept;
};

using ModelPtr   = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

}  // namespace unirt::llama_plugin

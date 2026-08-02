// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <llama.h>

#include "plugin/llm_backend.h"
#include "weight_cache.h"

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

/**
 * UniRT's text-generation contract implemented only through llama.cpp's
 * public API. An instance owns one context, one sequence, and its KV state;
 * the weights behind it are shared with any other handle on the same file.
 * Calls on the same handle are serialized because llama_context is mutable.
 */
class LlamaCppLlm final : public LlmBackend {
   public:
    LlamaCppLlm() = default;
    ~LlamaCppLlm() override = default;

    LlamaCppLlm(const LlamaCppLlm&)            = delete;
    LlamaCppLlm& operator=(const LlamaCppLlm&) = delete;

    int32_t create(const unirt_LlmCreateInput* input) override;
    int32_t reset() override;
    int32_t save_kv_cache(
        const unirt_KvCacheSaveInput* input, unirt_KvCacheSaveOutput* output) override;
    int32_t load_kv_cache(
        const unirt_KvCacheLoadInput* input, unirt_KvCacheLoadOutput* output) override;
    int32_t apply_chat_template(
        const unirt_LlmApplyChatTemplateInput* input,
        unirt_LlmApplyChatTemplateOutput* output) override;
    int32_t generate(
        const unirt_LlmGenerateInput* input, unirt_LlmGenerateOutput* output) override;
    int32_t get_model_info(unirt_LlmModelInfo* output) override;
    int32_t get_runtime_stats(unirt_LlmRuntimeStats* output) override;

   private:
    int32_t decode(const llama_token* tokens, int32_t count);
    int32_t tokenize(const char* text, bool add_special, std::vector<llama_token>& output) const;
    std::string token_piece(llama_token token) const;
    SamplerPtr make_sampler(const unirt_SamplerConfig* config, int32_t& error, bool& has_grammar) const;
    void clear_model() noexcept;

    // Drop the oldest cached tokens (past the pinned head) to make room for
    // `incoming` more. Returns how many were evicted; 0 = cannot shift.
    int32_t evict_for_space(int32_t incoming);

    // Longest shared prefix between the cached transcript and `wanted`,
    // capped so at least one token is left to re-evaluate for fresh logits.
    size_t reusable_prefix(const std::vector<llama_token>& wanted) const;

    mutable std::mutex mutex_;
    // Shared with every other handle that opened the same file on the same
    // device: llama_model is read-only, and it is context_ below that holds
    // this handle's own mutable KV state. See weight_cache.h.
    SharedModel       gguf_model_;
    ContextPtr        context_;
    const llama_vocab* vocab_ = nullptr;

    // llama_model_params::devices is an array terminated by nullptr. Keep it
    // alive for the model lifetime even though current llama.cpp consumes it
    // during model loading.
    std::vector<ggml_backend_dev_t> devices_;
    std::vector<llama_token>        history_;
    std::string                     template_override_;
    std::string                     device_name_ = "CPU";
    int32_t                         context_size_ = 0;
    // Context shifting: evict-and-continue when the window fills, keeping the
    // first `pinned_head_` tokens anchored. Disabled for models whose memory
    // cannot shift (recurrent / SWA-incompatible).
    bool    shift_supported_ = false;
    int32_t pinned_head_     = 4;
};

}  // namespace unirt::llama_plugin

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <llama.h>

#include "batch_engine.h"
#include "llama_ptr.h"
#include "plugin/llm_backend.h"
#include "weight_cache.h"

namespace unirt::llama_plugin {

/**
 * The small model whose guesses the real one checks.
 *
 * Speculative decoding trades arithmetic for latency: the draft proposes a
 * run of tokens one at a time (cheap), and the target verifies all of them in
 * a single batch (one pass instead of N). Everything the target agrees with is
 * free -- it would have produced the same tokens -- and everything it rejects
 * is work thrown away, which is why the proposal length is a knob and not a
 * constant. Greedy sampling only: a proposal is a guess at what the target
 * would most likely say, and there is nothing to gain from making the guess
 * itself random.
 *
 * Owned by the LlamaCppLlm it drafts for. Its KV is kept in step with the
 * accepted prefix by sync(), which is the only thing the target has to tell it.
 */
class DraftModel {
   public:
    int32_t load(const std::string& path, const llama_model_params& model_params,
                 const llama_context_params& context_params, const std::string& device_key,
                 const llama_vocab* target_vocab);

    bool ready() const { return context_ != nullptr; }

    // Make the draft's KV hold exactly `prefix`, reusing what it already has.
    int32_t sync(const std::vector<llama_token>& prefix);

    // Up to `count` greedily-drafted continuations of the synced prefix. Short
    // returns are normal: an EOG proposal ends the run, and so does a decode
    // that could not fit.
    std::vector<llama_token> propose(int32_t count);

   private:
    SharedModel              weights_;
    ContextPtr               context_;
    SamplerPtr               sampler_;
    const llama_vocab*       vocab_ = nullptr;
    std::vector<llama_token> history_;
    int32_t                  context_size_ = 0;
};

/**
 * UniRT's text-generation contract implemented only through llama.cpp's
 * public API. An instance owns one sequence of a BatchEngine and the KV state
 * behind it; the context and the weights are shared with any other handle
 * opened on the same file with the same geometry, so N handles decode in one
 * batch over one copy of the model. Calls on the same handle are serialized
 * because the sequence's transcript is mutable; calls on *different* handles
 * run concurrently and meet only inside the engine.
 */
class LlamaCppLlm final : public LlmBackend {
   public:
    LlamaCppLlm() = default;
    ~LlamaCppLlm() override;

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
    // Put `count` tokens into this sequence's KV, in batch-sized chunks and
    // batched with whatever other handles are decoding. Leaves last_logits_
    // pointing at the model's scores for the final token.
    int32_t decode(const llama_token* tokens, int32_t count);

    // One verification round: propose with the draft, check the proposals
    // against the target in a single batch, and keep the agreed prefix plus
    // the target's own next token. Appends what it kept to `accepted` and
    // leaves the target's KV holding exactly that. Returns UNIRT_SUCCESS, or
    // a decode error; `spoiled` says the batch could not be evaluated at all
    // and the caller should fall back to plain decoding for this step.
    int32_t speculate(
        llama_sampler* sampler, int32_t budget, std::vector<llama_token>& accepted,
        bool& spoiled);

    // Run the chain over one row of model scores without accepting the result.
    // llama_sampler_sample() accepts internally, which is right for a token
    // that is definitely being kept and wrong for a proposal that may be
    // rejected -- the rejected token would still have entered the penalty and
    // grammar state. It also insists on reading the context's own output
    // buffer, which a shared engine may already have decoded over.
    llama_token pick(llama_sampler* sampler, const float* logits) const;
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
    // The context and the weights are both shared; what belongs to this handle
    // is sequence_, an independent region of the KV cache. See batch_engine.h.
    SharedEngine       engine_;
    int32_t            sequence_ = BatchEngine::kNoSequence;
    const llama_vocab* vocab_    = nullptr;
    // Scores for the last token this sequence decoded, from the engine. Only
    // valid until the next submission on this handle.
    const float* last_logits_ = nullptr;

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
    // Per request: whether the caller asked for eviction instead of an
    // overflow error. Off unless asked, which is what the ABI promises.
    bool    sliding_window_  = false;
    int32_t pinned_head_     = 4;
    // Reused by pick(): a candidate array is one entry per vocabulary token,
    // and speculation builds one per verification position. Allocating that
    // per call was measurable next to the decode it is meant to save.
    mutable std::vector<llama_token_data> candidates_;
    // Optional, and off unless the caller passed draft_model_path.
    DraftModel draft_;
    int32_t    draft_tokens_ = 4;
    // Accepted and proposed over this handle's lifetime, for the acceptance
    // rate in the log. A draft model that is a poor match is otherwise silent:
    // it just makes everything slower.
    int64_t draft_proposed_ = 0;
    int64_t draft_accepted_ = 0;
};

}  // namespace unirt::llama_plugin

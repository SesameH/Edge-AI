// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <llama.h>
#include <mtmd.h>

#include "llm.h"
#include "plugin/vlm_backend.h"

namespace unirt::llama_plugin {

/**
 * Multimodal inference implemented with llama.cpp's public libmtmd API.
 *
 * Each handle owns one text model, llama context, multimodal projector, and
 * sequence.  The current UniRT VLM ABI supplies a full prompt on each call,
 * so n_past=0 is the supported and deterministic generation mode.
 */
class LlamaCppVlm final : public VlmBackend {
   public:
    LlamaCppVlm() = default;
    ~LlamaCppVlm() override = default;

    LlamaCppVlm(const LlamaCppVlm&)            = delete;
    LlamaCppVlm& operator=(const LlamaCppVlm&) = delete;

    int32_t create(const unirt_VlmCreateInput* input) override;
    int32_t reset() override;
    int32_t apply_chat_template(
        const unirt_VlmApplyChatTemplateInput* input,
        unirt_VlmApplyChatTemplateOutput* output) override;
    int32_t generate(
        const unirt_VlmGenerateInput* input,
        unirt_VlmGenerateOutput* output) override;
    int32_t get_capabilities(unirt_VlmCapabilities* output) override;

   private:
    enum class MediaKind { image, audio };

    void clear_model() noexcept;
    std::string token_piece(llama_token token) const;
    SamplerPtr make_sampler(const unirt_SamplerConfig* config, int32_t& error) const;
    int32_t decode_token(llama_token token, llama_pos position);

    mutable std::mutex mutex_;
    ModelPtr           model_;
    ContextPtr         context_;
    mtmd::context_ptr  multimodal_;
    const llama_vocab* vocab_ = nullptr;

    std::vector<ggml_backend_dev_t> devices_;
    std::string                     chat_template_;
    int32_t                         context_size_ = 0;
    int32_t                         batch_size_ = 0;
    llama_pos                       n_past_ = 0;
};

}  // namespace unirt::llama_plugin

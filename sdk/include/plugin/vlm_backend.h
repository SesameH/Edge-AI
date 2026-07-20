// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "unirt.h"

namespace unirt {

/**
 * Multimodal-generation contract a backend implements. Mirrors LlmBackend
 * with media-aware inputs; the same per-instance serialization rules apply.
 */
class VlmBackend {
   public:
    virtual ~VlmBackend() = default;

    /** Load the model (and its projector/encoder) described by `input`. */
    virtual int32_t create(const unirt_VlmCreateInput* input) = 0;

    /** Drop all decode state. */
    virtual int32_t reset() = 0;

    /** Render a multimodal message list through the model's template. */
    virtual int32_t apply_chat_template(
        const unirt_VlmApplyChatTemplateInput* input, unirt_VlmApplyChatTemplateOutput* output) = 0;

    virtual int32_t generate(const unirt_VlmGenerateInput* input, unirt_VlmGenerateOutput* output) = 0;

    /** Which media kinds this loaded model accepts. Default: none, so text-
     *  only backends stay honest without extra code. */
    virtual int32_t get_capabilities(unirt_VlmCapabilities* output) {
        if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
        output->supports_vision = false;
        output->supports_audio  = false;
        return UNIRT_SUCCESS;
    }
};

}  // namespace unirt

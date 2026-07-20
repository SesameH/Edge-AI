// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "unirt.h"

namespace unirt {

/**
 * Text-generation contract a backend implements. One instance owns one
 * loaded model plus its decode state (KV cache, transcript). The bridge
 * serializes calls per instance; implementations may add their own locking
 * for extra safety but must not assume cross-instance exclusion.
 */
class LlmBackend {
   public:
    virtual ~LlmBackend() = default;

    /** Load the model described by `input`. Called exactly once, first. */
    virtual int32_t create(const unirt_LlmCreateInput* input) = 0;

    /** Drop all decode state; the next generate starts a fresh transcript. */
    virtual int32_t reset() = 0;

    virtual int32_t save_kv_cache(const unirt_KvCacheSaveInput* input, unirt_KvCacheSaveOutput* output) = 0;
    virtual int32_t load_kv_cache(const unirt_KvCacheLoadInput* input, unirt_KvCacheLoadOutput* output) = 0;

    /** Render a message list through the model's chat template. */
    virtual int32_t apply_chat_template(
        const unirt_LlmApplyChatTemplateInput* input, unirt_LlmApplyChatTemplateOutput* output) = 0;

    /** Run one generation. Token pieces stream through input->on_token as
     *  raw bytes; UTF-8 reassembly happens in the bridge. */
    virtual int32_t generate(const unirt_LlmGenerateInput* input, unirt_LlmGenerateOutput* output) = 0;

    /** Static metadata (vocabulary size, BOS policy). Override when the
     *  backend can report it; the default keeps minimal backends building. */
    virtual int32_t get_model_info(unirt_LlmModelInfo*) { return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED; }

    /** Memory footprints (weights / KV / device peak). Leave unknown byte
     *  fields at -1; the bridge fills process-level numbers. */
    virtual int32_t get_runtime_stats(unirt_LlmRuntimeStats*) { return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED; }
};

}  // namespace unirt

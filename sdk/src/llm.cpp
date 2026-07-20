// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// unirt_llm_* C entrypoints: argument validation plus forwarding to the
// plugin's LlmBackend through the shared bridge machinery.

#include <cstring>

#include "bridge_support.h"
#include "plugin/llm_backend.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>

#include <fstream>
#endif

using namespace unirt;
using namespace unirt::bridge;

namespace {

bool llm_generation_input_sane(const unirt_LlmGenerateInput* input) noexcept {
    if (!input || input->input_ids_count < 0) return false;
    if (input->input_ids_count > 0 && !input->input_ids) return false;
    if (input->input_ids_count == 0 && !input->prompt_utf8) return false;
    if (!input->config) return true;
    if (!generation_config_sane(*input->config)) return false;
    // Media belongs to the VLM entrypoint. Silently ignoring it in an LLM
    // backend makes capability bugs extremely difficult to diagnose.
    return input->config->image_count == 0 && input->config->audio_count == 0;
}

int64_t resident_set_bytes() noexcept {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t      count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
        KERN_SUCCESS) {
        return static_cast<int64_t>(info.resident_size);
    }
    return -1;
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    long          total = 0, resident = 0;
    if (statm >> total >> resident) return static_cast<int64_t>(resident) * sysconf(_SC_PAGESIZE);
    return -1;
#else
    return -1;
#endif
}

}  // namespace

int32_t unirt_llm_create(const unirt_LlmCreateInput* input, unirt_LLM** out_handle) {
    if (!out_handle) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *out_handle = nullptr;
    if (!input || !input->plugin_id || !input->plugin_id[0] || !model_config_sane(input->config)) {
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    UNIRT_LOG_TRACE("{}", input);
    return open_backend<LlmBackend>("llm.create", input, out_handle);
}

int32_t unirt_llm_destroy(unirt_LLM* handle) {
    UNIRT_LOG_TRACE("llm.destroy");
    return close_backend<LlmBackend>("llm.destroy", handle);
}

int32_t unirt_llm_reset(unirt_LLM* handle) {
    UNIRT_LOG_TRACE("llm.reset");
    return with_backend<LlmBackend>(
        "llm.reset", handle, UNIRT_ERROR_COMMON_UNKNOWN, [](LlmBackend& llm) { return llm.reset(); });
}

int32_t unirt_llm_save_kv_cache(
    unirt_LLM* handle, const unirt_KvCacheSaveInput* input, unirt_KvCacheSaveOutput* output) {
    if (output) *output = {};
    if (!input || !input->path || !input->path[0]) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    UNIRT_LOG_TRACE("{}", input);
    return with_backend<LlmBackend>(
        "llm.save_kv_cache", handle, UNIRT_ERROR_COMMON_UNKNOWN,
        [&](LlmBackend& llm) { return llm.save_kv_cache(input, output); });
}

int32_t unirt_llm_load_kv_cache(
    unirt_LLM* handle, const unirt_KvCacheLoadInput* input, unirt_KvCacheLoadOutput* output) {
    if (output) *output = {};
    if (!input || !input->path || !input->path[0]) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    UNIRT_LOG_TRACE("{}", input);
    return with_backend<LlmBackend>(
        "llm.load_kv_cache", handle, UNIRT_ERROR_COMMON_UNKNOWN,
        [&](LlmBackend& llm) { return llm.load_kv_cache(input, output); });
}

int32_t unirt_llm_apply_chat_template(
    unirt_LLM* handle, const unirt_LlmApplyChatTemplateInput* input,
    unirt_LlmApplyChatTemplateOutput* output) {
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    output->formatted_text = nullptr;
    if (!input || input->message_count < 0 || (input->message_count > 0 && !input->messages)) {
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    UNIRT_LOG_TRACE("{}", input);
    return with_backend<LlmBackend>(
        "llm.apply_chat_template", handle, UNIRT_ERROR_COMMON_UNKNOWN,
        [&](LlmBackend& llm) { return llm.apply_chat_template(input, output); });
}

int32_t unirt_llm_generate(
    unirt_LLM* handle, const unirt_LlmGenerateInput* input, unirt_LlmGenerateOutput* output) {
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *output = {};
    if (!llm_generation_input_sane(input)) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    UNIRT_LOG_TRACE("{}", input);
    return with_backend<LlmBackend>(
        "llm.generate", handle, UNIRT_ERROR_COMMON_UNKNOWN,
        [&](LlmBackend& llm) { return run_generation(llm, input, output); });
}

int32_t unirt_llm_get_model_info(unirt_LLM* handle, unirt_LlmModelInfo* output) {
    UNIRT_LOG_TRACE("llm.get_model_info");
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    std::memset(output, 0, sizeof(*output));
    return with_backend<LlmBackend>(
        "llm.get_model_info", handle, UNIRT_ERROR_COMMON_UNKNOWN,
        [&](LlmBackend& llm) { return llm.get_model_info(output); });
}

int32_t unirt_llm_get_runtime_stats(unirt_LLM* handle, unirt_LlmRuntimeStats* output) {
    UNIRT_LOG_TRACE("llm.get_runtime_stats");
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    std::memset(output, 0, sizeof(*output));
    output->model_bytes       = -1;
    output->kv_cache_bytes    = -1;
    output->device_peak_bytes = -1;
    return with_backend<LlmBackend>(
        "llm.get_runtime_stats", handle, UNIRT_ERROR_COMMON_UNKNOWN, [&](LlmBackend& llm) {
            const int32_t rc          = llm.get_runtime_stats(output);
            output->process_rss_bytes = resident_set_bytes();
            return rc;
        });
}

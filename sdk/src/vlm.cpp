// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// unirt_vlm_* C entrypoints: argument validation plus forwarding to the
// plugin's VlmBackend through the shared bridge machinery.

#include "bridge_support.h"
#include "plugin/vlm_backend.h"

using namespace unirt;
using namespace unirt::bridge;

namespace {

bool template_input_sane(const unirt_VlmApplyChatTemplateInput* input) noexcept {
    if (!input || input->message_count < 0 || (input->message_count > 0 && !input->messages)) {
        return false;
    }
    for (int32_t i = 0; i < input->message_count; ++i) {
        const auto& message = input->messages[i];
        if (message.content_count < 0 || (message.content_count > 0 && !message.contents)) {
            return false;
        }
    }
    return true;
}

bool vlm_generation_input_sane(const unirt_VlmGenerateInput* input) noexcept {
    if (!input || !input->prompt_utf8) return false;
    if (!input->config) return true;
    if (!generation_config_sane(*input->config)) return false;
    for (int32_t i = 0; i < input->config->image_count; ++i) {
        if (!input->config->image_paths[i] || !input->config->image_paths[i][0]) return false;
    }
    for (int32_t i = 0; i < input->config->audio_count; ++i) {
        if (!input->config->audio_paths[i] || !input->config->audio_paths[i][0]) return false;
    }
    return true;
}

}  // namespace

int32_t unirt_vlm_create(const unirt_VlmCreateInput* input, unirt_VLM** out_handle) {
    if (!out_handle) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *out_handle = nullptr;
    if (!input || !input->plugin_id || !input->plugin_id[0] || !model_config_sane(input->config)) {
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    UNIRT_LOG_TRACE("{}", input);
    return open_backend<VlmBackend>("vlm.create", input, out_handle);
}

int32_t unirt_vlm_destroy(unirt_VLM* handle) {
    UNIRT_LOG_TRACE("vlm.destroy");
    return close_backend<VlmBackend>("vlm.destroy", handle);
}

int32_t unirt_vlm_reset(unirt_VLM* handle) {
    UNIRT_LOG_TRACE("vlm.reset");
    return with_backend<VlmBackend>(
        "vlm.reset", handle, UNIRT_ERROR_COMMON_UNKNOWN, [](VlmBackend& vlm) { return vlm.reset(); });
}

int32_t unirt_vlm_apply_chat_template(
    unirt_VLM* handle, const unirt_VlmApplyChatTemplateInput* input,
    unirt_VlmApplyChatTemplateOutput* output) {
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    output->formatted_text = nullptr;
    if (!template_input_sane(input)) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    UNIRT_LOG_TRACE("{}", input);
    return with_backend<VlmBackend>(
        "vlm.apply_chat_template", handle, UNIRT_ERROR_COMMON_UNKNOWN,
        [&](VlmBackend& vlm) { return vlm.apply_chat_template(input, output); });
}

int32_t unirt_vlm_get_capabilities(unirt_VLM* handle, unirt_VlmCapabilities* output) {
    UNIRT_LOG_TRACE("vlm.get_capabilities");
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *output = {};
    return with_backend<VlmBackend>(
        "vlm.get_capabilities", handle, UNIRT_ERROR_COMMON_UNKNOWN,
        [&](VlmBackend& vlm) { return vlm.get_capabilities(output); });
}

int32_t unirt_vlm_generate(
    unirt_VLM* handle, const unirt_VlmGenerateInput* input, unirt_VlmGenerateOutput* output) {
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *output = {};
    if (!vlm_generation_input_sane(input)) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    UNIRT_LOG_TRACE("{}", input);
    return with_backend<VlmBackend>(
        "vlm.generate", handle, UNIRT_ERROR_COMMON_UNKNOWN,
        [&](VlmBackend& vlm) { return run_generation(vlm, input, output); });
}

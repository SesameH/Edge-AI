// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>

#include "unirt.h"

namespace {

struct ErrorEntry {
    int32_t     code;
    const char* message;
};

// Table-driven so adding an error code is one line here plus the enum.
constexpr ErrorEntry kErrorTable[] = {
    {UNIRT_SUCCESS, "Success"},

    {UNIRT_ERROR_COMMON_UNKNOWN, "Unknown error"},
    {UNIRT_ERROR_COMMON_INVALID_INPUT, "Invalid input parameters or handle"},
    {UNIRT_ERROR_COMMON_INVALID_DEVICE, "Unknown device alias (expected one of: cpu, gpu, npu, hybrid)"},
    {UNIRT_ERROR_COMMON_MEMORY_ALLOCATION, "Memory allocation failed"},
    {UNIRT_ERROR_COMMON_FILE_NOT_FOUND, "File not found or inaccessible"},
    {UNIRT_ERROR_COMMON_NETWORK, "Network failure"},
    {UNIRT_ERROR_COMMON_CANCELLED, "Operation cancelled by caller"},
    {UNIRT_ERROR_COMMON_NOT_INITIALIZED, "Library not initialized"},
    {UNIRT_ERROR_COMMON_ALREADY_INITIALIZED, "Library already initialized"},
    {UNIRT_ERROR_COMMON_AUTH, "Hub rejected request; authentication required"},
    {UNIRT_ERROR_COMMON_HUB_MODEL_NOT_FOUND, "Model not found on hub"},
    {UNIRT_ERROR_COMMON_RATE_LIMITED, "Hub rate limit exceeded"},
    {UNIRT_ERROR_COMMON_HUB_SERVER, "Hub server error"},
    {UNIRT_ERROR_COMMON_NOT_SUPPORTED, "Operation not supported"},
    {UNIRT_ERROR_COMMON_MANIFEST_PARSE, "Manifest parse failed"},
    {UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED, "Parameter not supported by this plugin"},
    {UNIRT_ERROR_COMMON_BUSY, "Runtime is busy with active model handles"},

    {UNIRT_ERROR_COMMON_MODEL_LOAD, "Model loading failed"},
    {UNIRT_ERROR_COMMON_MODEL_INVALID, "Invalid model format"},

    {UNIRT_ERROR_COMMON_PLUGIN_LOAD, "Plugin loading failed"},
    {UNIRT_ERROR_COMMON_PLUGIN_INVALID, "Invalid plugin"},

    {UNIRT_ERROR_LLM_TOKENIZATION_FAILED, "Tokenization failed"},
    {UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH, "Context length exceeded"},
    {UNIRT_ERROR_LLM_GENERATION_FAILED, "Text generation failed"},
    {UNIRT_ERROR_LLM_GENERATION_PROMPT_TOO_LONG, "Input prompt too long"},

    {UNIRT_ERROR_VLM_IMAGE_LOAD, "Image loading failed"},
    {UNIRT_ERROR_VLM_IMAGE_FORMAT, "Unsupported image format"},
    {UNIRT_ERROR_VLM_AUDIO_LOAD, "Audio loading failed"},
    {UNIRT_ERROR_VLM_AUDIO_FORMAT, "Unsupported audio format"},
    {UNIRT_ERROR_VLM_GENERATION_FAILED, "Multimodal generation failed"},

    {UNIRT_ERROR_EMBEDDING_INFERENCE_FAILED, "Embedding inference failed"},
    {UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID, "Embedding model output is invalid"},
};

}  // namespace

const char* unirt_get_error_message(const unirt_ErrorCode error_code) {
    for (const auto& entry : kErrorTable) {
        if (entry.code == error_code) return entry.message;
    }
    return "Unknown error code";
}

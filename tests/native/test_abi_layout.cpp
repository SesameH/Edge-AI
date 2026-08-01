// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause
//
// Dump the memory layout of every ABI struct the Python binding mirrors.
//
// unirt.h and bindings/python/unirt/_ffi/_types.py describe the same bytes in
// two languages, kept in step by hand. Nothing checked that until this: a field
// appended to one and forgotten in the other, or added in a place where padding
// lands differently, produces no compiler error and no import error -- just a
// binding that reads the wrong offset and passes garbage across the boundary.
//
// Prints JSON to stdout; tests/python/test_abi_layout.py compares it to what
// ctypes computes for the same structs.

#include <cstddef>
#include <cstdio>

#include "unirt.h"

namespace {

bool first_struct = true;
bool first_field  = true;

void begin_struct(const char* name, size_t size, size_t alignment) {
    std::printf("%s\n  \"%s\": {\"size\": %zu, \"align\": %zu, \"fields\": {",
                first_struct ? "" : ",", name, size, alignment);
    first_struct = false;
    first_field  = true;
}

void field(const char* name, size_t offset, size_t size) {
    std::printf("%s\"%s\": [%zu, %zu]", first_field ? "" : ", ", name, offset, size);
    first_field = false;
}

void end_struct() { std::printf("}}"); }

}  // namespace

#define BEGIN(type) begin_struct(#type, sizeof(type), alignof(type))
#define FIELD(type, member) field(#member, offsetof(type, member), sizeof(((type*)nullptr)->member))

int main() {
    std::printf("{");

    BEGIN(unirt_Logprob);
    FIELD(unirt_Logprob, piece);
    FIELD(unirt_Logprob, token_id);
    FIELD(unirt_Logprob, logprob);
    end_struct();

    BEGIN(unirt_ProfileData);
    FIELD(unirt_ProfileData, ttft);
    FIELD(unirt_ProfileData, prompt_time);
    FIELD(unirt_ProfileData, decode_time);
    FIELD(unirt_ProfileData, prompt_tokens);
    FIELD(unirt_ProfileData, generated_tokens);
    FIELD(unirt_ProfileData, audio_duration);
    FIELD(unirt_ProfileData, prefill_speed);
    FIELD(unirt_ProfileData, decoding_speed);
    FIELD(unirt_ProfileData, real_time_factor);
    FIELD(unirt_ProfileData, stop_reason);
    end_struct();

    BEGIN(unirt_SamplerConfig);
    FIELD(unirt_SamplerConfig, seed);
    FIELD(unirt_SamplerConfig, temperature);
    FIELD(unirt_SamplerConfig, top_k);
    FIELD(unirt_SamplerConfig, top_p);
    FIELD(unirt_SamplerConfig, min_p);
    FIELD(unirt_SamplerConfig, repetition_penalty);
    FIELD(unirt_SamplerConfig, presence_penalty);
    FIELD(unirt_SamplerConfig, frequency_penalty);
    FIELD(unirt_SamplerConfig, grammar_path);
    FIELD(unirt_SamplerConfig, grammar_string);
    FIELD(unirt_SamplerConfig, json_schema);
    FIELD(unirt_SamplerConfig, enable_json);
    end_struct();

    BEGIN(unirt_GenerationConfig);
    FIELD(unirt_GenerationConfig, max_tokens);
    FIELD(unirt_GenerationConfig, stop);
    FIELD(unirt_GenerationConfig, stop_count);
    FIELD(unirt_GenerationConfig, n_past);
    FIELD(unirt_GenerationConfig, sampler_config);
    FIELD(unirt_GenerationConfig, image_paths);
    FIELD(unirt_GenerationConfig, image_count);
    FIELD(unirt_GenerationConfig, image_max_length);
    FIELD(unirt_GenerationConfig, audio_paths);
    FIELD(unirt_GenerationConfig, audio_count);
    FIELD(unirt_GenerationConfig, sliding_window);
    FIELD(unirt_GenerationConfig, sliding_window_n_keep);
    FIELD(unirt_GenerationConfig, logprobs);
    end_struct();

    BEGIN(unirt_ModelConfig);
    FIELD(unirt_ModelConfig, n_ctx);
    FIELD(unirt_ModelConfig, n_threads);
    FIELD(unirt_ModelConfig, n_threads_batch);
    FIELD(unirt_ModelConfig, n_batch);
    FIELD(unirt_ModelConfig, n_ubatch);
    FIELD(unirt_ModelConfig, n_seq_max);
    FIELD(unirt_ModelConfig, n_gpu_layers);
    FIELD(unirt_ModelConfig, chat_template_path);
    FIELD(unirt_ModelConfig, chat_template_content);
    FIELD(unirt_ModelConfig, grammar_str);
    end_struct();

    BEGIN(unirt_LlmCreateInput);
    FIELD(unirt_LlmCreateInput, model_path);
    FIELD(unirt_LlmCreateInput, tokenizer_path);
    FIELD(unirt_LlmCreateInput, config);
    FIELD(unirt_LlmCreateInput, plugin_id);
    FIELD(unirt_LlmCreateInput, device_id);
    end_struct();

    BEGIN(unirt_LlmGenerateInput);
    FIELD(unirt_LlmGenerateInput, prompt_utf8);
    FIELD(unirt_LlmGenerateInput, config);
    FIELD(unirt_LlmGenerateInput, on_token);
    FIELD(unirt_LlmGenerateInput, user_data);
    FIELD(unirt_LlmGenerateInput, input_ids);
    FIELD(unirt_LlmGenerateInput, input_ids_count);
    FIELD(unirt_LlmGenerateInput, on_logprob);
    end_struct();

    BEGIN(unirt_LlmGenerateOutput);
    FIELD(unirt_LlmGenerateOutput, full_text);
    FIELD(unirt_LlmGenerateOutput, profile_data);
    end_struct();

    BEGIN(unirt_LlmModelInfo);
    FIELD(unirt_LlmModelInfo, vocab_size);
    FIELD(unirt_LlmModelInfo, bos_token);
    FIELD(unirt_LlmModelInfo, add_bos);
    FIELD(unirt_LlmModelInfo, reserved0);
    end_struct();

    BEGIN(unirt_LlmRuntimeStats);
    FIELD(unirt_LlmRuntimeStats, model_bytes);
    FIELD(unirt_LlmRuntimeStats, kv_cache_bytes);
    FIELD(unirt_LlmRuntimeStats, device_peak_bytes);
    FIELD(unirt_LlmRuntimeStats, process_rss_bytes);
    FIELD(unirt_LlmRuntimeStats, device_name);
    end_struct();

    std::printf("\n}\n");
    return 0;
}

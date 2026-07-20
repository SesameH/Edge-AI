// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// End-to-end smoke at the C API level: load a GGUF, render the chat
// template, generate greedily, and require the streamed pieces to reassemble
// into the returned text. Runs anywhere a llama_cpp backend exists — on iOS
// simulators via `simctl spawn` (static build), on desktops via ctest.
// Exit 77 = skip (model not downloaded).

#include <cstdio>
#include <cstring>
#include <string>

#include "unirt.h"

#if defined(UNIRT_SMOKE_STATIC_PLUGIN)
extern "C" unirt_PluginId unirt_plugin_id(void);
extern "C" unirt_PluginTable* unirt_plugin_open(void);
#endif

namespace {

bool collect(const char* piece, void* user_data) {
    static_cast<std::string*>(user_data)->append(piece ? piece : "");
    return true;
}

#define REQUIRE(condition)                                                        \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::fprintf(                                                         \
                stderr, "FAILED: %s (line %d): %s\n", #condition, __LINE__,       \
                unirt_last_error_message());                                      \
            return 1;                                                             \
        }                                                                         \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 77;
    }
    if (FILE* probe = std::fopen(argv[1], "rb")) {
        std::fclose(probe);
    } else {
        std::fprintf(stderr, "SKIP: test model not present: %s\n", argv[1]);
        return 77;
    }

#if defined(UNIRT_SMOKE_STATIC_PLUGIN)
    REQUIRE(unirt_register_plugin(unirt_plugin_id, unirt_plugin_open) == UNIRT_SUCCESS);
#endif
    REQUIRE(unirt_init() == UNIRT_SUCCESS);

    unirt_LlmCreateInput create{};
    create.model_path          = argv[1];
    create.plugin_id           = "llama_cpp";
    create.config.n_ctx        = 512;
    create.config.n_gpu_layers = 0;  // CPU: identical behavior on every host

    unirt_LLM* llm = nullptr;
    REQUIRE(unirt_llm_create(&create, &llm) == UNIRT_SUCCESS);

    unirt_LlmChatMessage message{
        "user", "What is the capital of France? Answer in one short sentence."};
    unirt_LlmApplyChatTemplateInput template_input{};
    template_input.messages              = &message;
    template_input.message_count         = 1;
    template_input.add_generation_prompt = true;
    unirt_LlmApplyChatTemplateOutput template_output{};
    REQUIRE(unirt_llm_apply_chat_template(llm, &template_input, &template_output) == UNIRT_SUCCESS);

    std::string            streamed;
    unirt_SamplerConfig   sampler{};  // zeroed = greedy
    unirt_GenerationConfig config{};
    config.max_tokens     = 32;
    config.sampler_config = &sampler;
    unirt_LlmGenerateInput generate_input{};
    generate_input.prompt_utf8 = template_output.formatted_text;
    generate_input.config      = &config;
    generate_input.on_token    = collect;
    generate_input.user_data   = &streamed;

    unirt_LlmGenerateOutput generate_output{};
    const int32_t status = unirt_llm_generate(llm, &generate_input, &generate_output);
    unirt_free(template_output.formatted_text);
    REQUIRE(status == UNIRT_SUCCESS);

    const char* full = static_cast<const char*>(generate_output.full_text);
    std::printf("reply: %s\n", full ? full : "(null)");
    const bool non_empty  = full && full[0] != '\0';
    const bool reassembles = full && streamed == full;
    unirt_free(generate_output.full_text);

    REQUIRE(unirt_llm_destroy(llm) == UNIRT_SUCCESS);
    REQUIRE(unirt_deinit() == UNIRT_SUCCESS);
    REQUIRE(non_empty);
    REQUIRE(reassembles);

    std::puts("native inference smoke: OK");
    return 0;
}

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// Several handles working one shared context as hard as they can, so a
// sanitizer has something to look at.
//
// Batching moved the plugin from "one context per handle, one thread at a
// time" to "one context, N threads, meeting inside llama_decode", and the
// interesting failures there are races rather than wrong answers: a transcript
// read while its owner rewrites it, a logits row overwritten under a reader, a
// sequence claimed while somebody else's round is in flight. None of those are
// reliably visible from the outside -- they corrupt a reply that still looks
// like a reply.
//
// So this is not an assertion suite. It is the workload to run under
// ThreadSanitizer, which is why it is a native binary: on macOS the sanitizer
// runtime cannot be loaded into the system Python, so the same threads driven
// from the Python bindings prove nothing.
//
//   cmake -S sdk -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
//         -DUNIRT_PLUGIN_MLX=OFF -DUNIRT_PLUGIN_ONNXRUNTIME=OFF -DGGML_METAL=OFF \
//         -DCMAKE_C_FLAGS=-fsanitize=thread -DCMAKE_CXX_FLAGS=-fsanitize=thread \
//         -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread \
//         -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread
//   cmake --build build-tsan -j8
//   ./build-tsan/unirt-batching-stress-test models/SmolLM2-135M-Instruct-Q8_0.gguf
//
// Exit 77 = skip (model not downloaded).

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "unirt.h"

namespace {

constexpr int32_t kSlots = 4;

// Long enough to be worth borrowing, so the prefix-sharing path runs too.
std::string shared_system() {
    std::string text = "You are terse.";
    for (int index = 0; index < 60; ++index) {
        text += " Background fact " + std::to_string(index) + ": the value is " +
                std::to_string(index) + ".";
    }
    return text;
}

const char* kQuestions[] = {
    "Name a colour.", "Name a fruit.", "Count to five.",
    "Name a country.", "Say hello.",   "Name an animal.",
};

bool collect(const char* piece, void* user_data) {
    static_cast<std::string*>(user_data)->append(piece ? piece : "");
    return true;
}

std::atomic<bool> stop{false};
std::atomic<int>  failures{0};

void fail(const char* what, int32_t code) {
    std::fprintf(stderr, "FAILED: %s -> %d: %s\n", what, code, unirt_last_error_message());
    failures.fetch_add(1);
}

unirt_LLM* open_handle(const char* path) {
    unirt_LlmCreateInput create{};
    create.model_path          = path;
    create.plugin_id           = "llama_cpp";
    create.config.n_ctx        = 1024;
    create.config.n_seq_max    = kSlots;
    create.config.n_gpu_layers = 0;
    unirt_LLM*    handle = nullptr;
    const int32_t code   = unirt_llm_create(&create, &handle);
    if (code != UNIRT_SUCCESS) {
        fail("unirt_llm_create", code);
        return nullptr;
    }
    return handle;
}

// Render <system?> + question through the model's own template.
char* render(unirt_LLM* handle, const std::string& system, const char* question) {
    unirt_LlmChatMessage messages[2];
    int32_t              count = 0;
    if (!system.empty()) messages[count++] = unirt_LlmChatMessage{"system", system.c_str()};
    messages[count++] = unirt_LlmChatMessage{"user", question};

    unirt_LlmApplyChatTemplateInput input{};
    input.messages              = messages;
    input.message_count         = count;
    input.add_generation_prompt = true;
    unirt_LlmApplyChatTemplateOutput output{};
    if (unirt_llm_apply_chat_template(handle, &input, &output) != UNIRT_SUCCESS) return nullptr;
    return output.formatted_text;
}

void one_generation(unirt_LLM* handle, const char* prompt, int32_t max_tokens,
                    bool sliding, int32_t n_past) {
    std::string            streamed;
    unirt_SamplerConfig    sampler{};
    unirt_GenerationConfig config{};
    config.max_tokens     = max_tokens;
    config.sampler_config = &sampler;
    config.sliding_window = sliding;
    config.n_past         = n_past;

    unirt_LlmGenerateInput input{};
    input.prompt_utf8 = prompt;
    input.config      = &config;
    input.on_token    = collect;
    input.user_data   = &streamed;

    unirt_LlmGenerateOutput output{};
    const int32_t code = unirt_llm_generate(handle, &input, &output);
    if (code == UNIRT_SUCCESS) {
        const char* full = static_cast<const char*>(output.full_text);
        if (full && streamed != full) {
            std::fprintf(stderr, "FAILED: streamed text differs from the returned text\n");
            failures.fetch_add(1);
        }
        unirt_free(output.full_text);
    } else if (n_past == 0) {
        // A deliberately invalid n_past is expected to fail; nothing else is.
        fail("unirt_llm_generate", code);
    }
}

void worker(unirt_LLM* handle, const std::string& system, unsigned seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    std::uniform_int_distribution<int>     tokens(4, 40);
    std::uniform_int_distribution<int>     question(0, 5);

    while (!stop.load()) {
        const double choice = pick(rng);
        char* prompt = render(handle, pick(rng) < 0.6 ? system : std::string(),
                              kQuestions[question(rng)]);
        if (!prompt) {
            fail("unirt_llm_apply_chat_template", -1);
            return;
        }
        if (choice < 0.40) {
            one_generation(handle, prompt, tokens(rng), false, 0);
        } else if (choice < 0.55) {
            // Runs past the window, so eviction runs beside everyone else.
            one_generation(handle, prompt, 200, true, 0);
        } else if (choice < 0.65) {
            unirt_llm_reset(handle);
        } else if (choice < 0.75) {
            unirt_LlmRuntimeStats stats{};
            unirt_llm_get_runtime_stats(handle, &stats);
        } else if (choice < 0.85) {
            // Expected to be rejected; the point is that it is rejected while
            // three other sequences are mid-round.
            one_generation(handle, prompt, 8, false, 2000000000);
        } else {
            unirt_LlmModelInfo info{};
            unirt_llm_get_model_info(handle, &info);
            one_generation(handle, prompt, tokens(rng), false, 0);
        }
        unirt_free(prompt);
    }
}

// Handles opening and closing while the others are working: the one path that
// mixes claim/release with a round in flight.
void churn(const char* path, const std::string& system, unsigned seed) {
    std::mt19937                       rng(seed);
    std::uniform_int_distribution<int> nap(40, 250);
    while (!stop.load()) {
        unirt_LLM* handle = open_handle(path);
        if (!handle) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(nap(rng)));
        if (char* prompt = render(handle, system, kQuestions[0])) {
            one_generation(handle, prompt, 8, false, 0);
            unirt_free(prompt);
        }
        if (unirt_llm_destroy(handle) != UNIRT_SUCCESS) fail("unirt_llm_destroy", -1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [seconds]\n", argv[0]);
        return 77;
    }
    if (FILE* probe = std::fopen(argv[1], "rb")) {
        std::fclose(probe);
    } else {
        std::fprintf(stderr, "SKIP: test model not present: %s\n", argv[1]);
        return 77;
    }
    const int seconds = argc > 2 ? std::atoi(argv[2]) : 15;

    if (unirt_init() != UNIRT_SUCCESS) {
        fail("unirt_init", -1);
        return 1;
    }

    const std::string system = shared_system();
    std::vector<unirt_LLM*> handles;
    for (int32_t slot = 0; slot < kSlots; ++slot) {
        unirt_LLM* handle = open_handle(argv[1]);
        if (!handle) return 1;
        handles.push_back(handle);
    }

    std::vector<std::thread> threads;
    for (int32_t slot = 0; slot < kSlots; ++slot) {
        threads.emplace_back(worker, handles[slot], std::cref(system),
                             static_cast<unsigned>(slot + 1));
    }
    threads.emplace_back(churn, argv[1], std::cref(system), 99u);

    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    stop.store(true);
    for (auto& thread : threads) thread.join();

    for (unirt_LLM* handle : handles) {
        if (unirt_llm_destroy(handle) != UNIRT_SUCCESS) fail("unirt_llm_destroy", -1);
    }
    if (unirt_deinit() != UNIRT_SUCCESS) fail("unirt_deinit", -1);

    if (failures.load() > 0) {
        std::fprintf(stderr, "batching stress: %d failures\n", failures.load());
        return 1;
    }
    std::puts("native batching stress: OK");
    return 0;
}

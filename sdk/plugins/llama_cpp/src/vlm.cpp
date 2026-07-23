// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "vlm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string_view>
#include <utility>

#include <mtmd-helper.h>

#include "generation_state.h"
#include "logging.h"
#include "device_label.h"
#include "schema_grammar.h"

namespace unirt::llama_plugin {

namespace {

constexpr const char* kJsonGrammar = R"gbnf(
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [0-9]{0,15})? ws
ws     ::= | " " | "\n" [ \t]{0,20}
)gbnf";

// The public generation ABI carries image and audio paths in separate arrays,
// while libmtmd uses one generic marker stream. Keep each formatted prompt
// self-describing so mixed-modality ordering does not depend on mutable state
// left by a previous apply_chat_template() call on the same handle. These tags
// are removed before mtmd_tokenize() sees the prompt.
constexpr std::string_view kImageMediaTag = "<__unirt_image__>";
constexpr std::string_view kAudioMediaTag = "<__unirt_audio__>";

char* copy_to_c(const std::string& value) {
    auto* output = static_cast<char*>(std::malloc(value.size() + 1));
    if (!output) return nullptr;
    std::memcpy(output, value.data(), value.size());
    output[value.size()] = '\0';
    return output;
}

int64_t monotonic_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool read_text_file(const char* path, std::string& output) {
    if (!path || !path[0]) return false;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    stream.seekg(0, std::ios::end);
    const auto length = stream.tellg();
    if (length < 0 || length > 4 * 1024 * 1024) return false;
    stream.seekg(0, std::ios::beg);
    output.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return stream.good() || stream.eof();
}

bool readable_file(const char* path) {
    if (!path || !path[0]) return false;
    std::ifstream stream(path, std::ios::binary);
    return stream.good();
}

bool valid_sampler_config(const unirt_SamplerConfig& config) {
    return config.top_k >= 0 && std::isfinite(config.temperature) &&
           config.temperature >= 0.0f && std::isfinite(config.top_p) &&
           config.top_p >= 0.0f && config.top_p <= 1.0f &&
           std::isfinite(config.min_p) && config.min_p >= 0.0f &&
           config.min_p <= 1.0f && std::isfinite(config.repetition_penalty) &&
           config.repetition_penalty >= 0.0f &&
           std::isfinite(config.presence_penalty) &&
           std::isfinite(config.frequency_penalty);
}

size_t count_markers(const std::string& prompt, const std::string& marker) {
    if (marker.empty()) return 0;
    size_t count = 0;
    size_t offset = 0;
    while ((offset = prompt.find(marker, offset)) != std::string::npos) {
        ++count;
        offset += marker.size();
    }
    return count;
}

struct BatchGuard {
    llama_batch batch{};

    BatchGuard() : batch(llama_batch_init(1, 0, 1)) {}
    ~BatchGuard() { llama_batch_free(batch); }

    BatchGuard(const BatchGuard&) = delete;
    BatchGuard& operator=(const BatchGuard&) = delete;
};

}  // namespace

void LlamaCppVlm::clear_model() noexcept {
    // mtmd retains a pointer to the text model, so it must be released first.
    multimodal_.reset();
    context_.reset();
    model_.reset();
    vocab_ = nullptr;
    devices_.clear();
    chat_template_.clear();
    device_name_ = "CPU";
    context_size_ = 0;
    batch_size_ = 0;
    n_past_ = 0;
}

int32_t LlamaCppVlm::create(const unirt_VlmCreateInput* input) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!input || !input->model_path || !input->model_path[0] ||
        !input->mmproj_path || !input->mmproj_path[0]) {
        UNIRT_LOG_ERROR("llama_cpp VLM: model_path and mmproj_path are required");
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    if (!readable_file(input->model_path) || !readable_file(input->mmproj_path)) {
        UNIRT_LOG_ERROR("llama_cpp VLM: model or mmproj file is not readable");
        return UNIRT_ERROR_COMMON_FILE_NOT_FOUND;
    }

    std::string requested_template;
    if (input->config.chat_template_content && input->config.chat_template_content[0]) {
        requested_template = input->config.chat_template_content;
    } else if (input->config.chat_template_path && input->config.chat_template_path[0] &&
               !read_text_file(input->config.chat_template_path, requested_template)) {
        return UNIRT_ERROR_COMMON_FILE_NOT_FOUND;
    }

    std::vector<ggml_backend_dev_t> selected_devices;
    std::string selected_device_name = "CPU";
    bool use_accelerator = input->config.n_gpu_layers != 0;
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = input->config.n_gpu_layers;

    if (input->device_id && input->device_id[0]) {
        ggml_backend_dev_t device = ggml_backend_dev_by_name(input->device_id);
        if (!device) return UNIRT_ERROR_COMMON_INVALID_DEVICE;
        selected_devices = {device, nullptr};
        model_params.devices = selected_devices.data();
        use_accelerator = ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                          model_params.n_gpu_layers != 0;
        selected_device_name = device_label(device, input->device_id);
    } else if (model_params.n_gpu_layers == 0) {
        if (auto* cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            selected_devices = {cpu, nullptr};
            model_params.devices = selected_devices.data();
            selected_device_name = device_label(cpu, "CPU");
        }
        use_accelerator = false;
    } else if (auto* gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        selected_device_name = device_label(gpu, "GPU");
    }

    ModelPtr model(llama_model_load_from_file(input->model_path, model_params));
    if (!model) return UNIRT_ERROR_COMMON_MODEL_LOAD;
    if (llama_model_has_encoder(model.get())) return UNIRT_ERROR_COMMON_MODEL_INVALID;

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = input->config.n_ctx > 0
                               ? static_cast<uint32_t>(input->config.n_ctx)
                               : 0;
    if (input->config.n_batch > 0) {
        context_params.n_batch = static_cast<uint32_t>(input->config.n_batch);
    }
    if (input->config.n_ubatch > 0) {
        context_params.n_ubatch = static_cast<uint32_t>(input->config.n_ubatch);
    }
    if (input->config.n_seq_max > 0) {
        context_params.n_seq_max = static_cast<uint32_t>(input->config.n_seq_max);
    }
    if (input->config.n_threads > 0) context_params.n_threads = input->config.n_threads;
    if (input->config.n_threads_batch > 0) {
        context_params.n_threads_batch = input->config.n_threads_batch;
    }
    context_params.no_perf = false;
    if (!use_accelerator) {
        context_params.offload_kqv = false;
        context_params.op_offload = false;
    }

    ContextPtr context(llama_init_from_model(model.get(), context_params));
    if (!context && (!input->device_id || !input->device_id[0]) &&
        model_params.n_gpu_layers != 0) {
        UNIRT_LOG_WARN("llama_cpp VLM: accelerator failed; retrying text model on CPU");
        model.reset();
        model_params.n_gpu_layers = 0;
        if (auto* cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            selected_devices = {cpu, nullptr};
            model_params.devices = selected_devices.data();
            selected_device_name = device_label(cpu, "CPU");
        } else {
            model_params.devices = nullptr;
        }
        use_accelerator = false;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        model.reset(llama_model_load_from_file(input->model_path, model_params));
        if (model) context.reset(llama_init_from_model(model.get(), context_params));
    }
    if (!context) return UNIRT_ERROR_COMMON_MODEL_LOAD;

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = use_accelerator;
    mtmd_params.print_timings = false;
    if (input->config.n_threads > 0) mtmd_params.n_threads = input->config.n_threads;
    mtmd::context_ptr multimodal(
        mtmd_init_from_file(input->mmproj_path, model.get(), mtmd_params));
    if (!multimodal && use_accelerator) {
        UNIRT_LOG_WARN("llama_cpp VLM: mmproj accelerator failed; retrying on CPU");
        mtmd_params.use_gpu = false;
        multimodal.reset(mtmd_init_from_file(input->mmproj_path, model.get(), mtmd_params));
    }
    if (!multimodal) return UNIRT_ERROR_COMMON_MODEL_LOAD;
    if (!mtmd_support_vision(multimodal.get()) && !mtmd_support_audio(multimodal.get())) {
        UNIRT_LOG_ERROR("llama_cpp VLM: mmproj reports no supported input modality");
        return UNIRT_ERROR_COMMON_MODEL_INVALID;
    }

    const int32_t actual_context = static_cast<int32_t>(llama_n_ctx(context.get()));
    const int32_t actual_batch = static_cast<int32_t>(llama_n_batch(context.get()));
    if (actual_context <= 0 || actual_batch <= 0) return UNIRT_ERROR_COMMON_MODEL_LOAD;

    clear_model();
    model_ = std::move(model);
    context_ = std::move(context);
    multimodal_ = std::move(multimodal);
    vocab_ = llama_model_get_vocab(model_.get());
    devices_ = std::move(selected_devices);
    chat_template_ = std::move(requested_template);
    device_name_ = std::move(selected_device_name);
    context_size_ = actual_context;
    batch_size_ = actual_batch;

    UNIRT_LOG_INFO(
        "llama_cpp VLM: loaded model={}, mmproj={}, context={}, batch={}, device={}",
        input->model_path, input->mmproj_path, context_size_, batch_size_,
        device_name_);
    return UNIRT_SUCCESS;
}

int32_t LlamaCppVlm::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!context_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    llama_memory_clear(llama_get_memory(context_.get()), true);
    llama_perf_context_reset(context_.get());
    n_past_ = 0;
    return UNIRT_SUCCESS;
}

int32_t LlamaCppVlm::get_capabilities(unirt_VlmCapabilities* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!multimodal_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    output->supports_vision = mtmd_support_vision(multimodal_.get());
    output->supports_audio = mtmd_support_audio(multimodal_.get());
    return UNIRT_SUCCESS;
}

int32_t LlamaCppVlm::get_runtime_stats(unirt_VlmRuntimeStats* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    // mtmd's public API has no projector-size accessor, so this covers only
    // the text model's weights, same as the LLM backend — not the mmproj.
    output->model_bytes = static_cast<int64_t>(llama_model_size(model_.get()));
    output->kv_cache_bytes = context_
                                  ? static_cast<int64_t>(llama_state_get_size(context_.get()))
                                  : -1;
    output->device_peak_bytes = -1;
    output->device_name = device_name_.c_str();
    return UNIRT_SUCCESS;
}

int32_t LlamaCppVlm::apply_chat_template(
    const unirt_VlmApplyChatTemplateInput* input,
    unirt_VlmApplyChatTemplateOutput* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_ || !multimodal_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output || input->message_count < 0 ||
        (input->message_count > 0 && !input->messages)) {
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    output->formatted_text = nullptr;
    if (input->grounding) {
        UNIRT_LOG_ERROR("llama_cpp VLM: grounding mode is not exposed by libmtmd");
        return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    if (input->tools && input->tools[0]) {
        UNIRT_LOG_ERROR("llama_cpp VLM: tool templates are not exposed by libmtmd");
        return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }

    const char* marker_ptr = mtmd_get_marker(multimodal_.get());
    const std::string marker = marker_ptr && marker_ptr[0]
                                   ? marker_ptr
                                   : mtmd_default_marker();
    std::vector<std::string> contents;
    std::vector<llama_chat_message> messages;
    contents.reserve(static_cast<size_t>(input->message_count));
    messages.reserve(static_cast<size_t>(input->message_count));
    size_t estimated_size = 256;

    for (int32_t index = 0; index < input->message_count; ++index) {
        const auto& source = input->messages[index];
        if (source.content_count < 0 ||
            (source.content_count > 0 && !source.contents)) {
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        std::string flattened;
        for (int64_t item_index = 0; item_index < source.content_count; ++item_index) {
            const auto& item = source.contents[item_index];
            if (!item.type || !item.type[0]) return UNIRT_ERROR_COMMON_INVALID_INPUT;
            if (std::strcmp(item.type, "text") == 0) {
                if (item.text) flattened += item.text;
            } else if (std::strcmp(item.type, "image") == 0 ||
                       std::strcmp(item.type, "image_url") == 0) {
                if (!mtmd_support_vision(multimodal_.get())) {
                    return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
                }
                flattened += marker;
                flattened += kImageMediaTag;
            } else if (std::strcmp(item.type, "audio") == 0 ||
                       std::strcmp(item.type, "input_audio") == 0) {
                if (!mtmd_support_audio(multimodal_.get())) {
                    return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
                }
                flattened += marker;
                flattened += kAudioMediaTag;
            } else {
                UNIRT_LOG_ERROR("llama_cpp VLM: unsupported content type '{}'", item.type);
                return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
            }
        }
        contents.push_back(std::move(flattened));
        estimated_size += contents.back().size() + 64;
    }
    for (int32_t index = 0; index < input->message_count; ++index) {
        const char* role = input->messages[index].role
                               ? input->messages[index].role
                               : "user";
        messages.push_back({role, contents[static_cast<size_t>(index)].c_str()});
        estimated_size += std::strlen(role);
    }
    estimated_size = std::min<size_t>(
        std::max<size_t>(estimated_size * 2, 512),
        static_cast<size_t>(std::numeric_limits<int32_t>::max()));

    const char* tmpl = chat_template_.empty()
                           ? llama_model_chat_template(model_.get(), nullptr)
                           : chat_template_.c_str();
    std::vector<char> buffer(estimated_size);
    // The current C ABI has no VLM add_generation_prompt field. VLM template
    // calls are generation-oriented, so end with the assistant prefix.
    int32_t length = llama_chat_apply_template(
        tmpl, messages.data(), messages.size(), true, buffer.data(),
        static_cast<int32_t>(buffer.size()));
    if (length > static_cast<int32_t>(buffer.size())) {
        buffer.resize(static_cast<size_t>(length));
        length = llama_chat_apply_template(
            tmpl, messages.data(), messages.size(), true, buffer.data(),
            static_cast<int32_t>(buffer.size()));
    }
    if (length < 0 || length > static_cast<int32_t>(buffer.size())) {
        return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }

    output->formatted_text = copy_to_c(
        std::string(buffer.data(), static_cast<size_t>(length)));
    if (!output->formatted_text) return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
    return UNIRT_SUCCESS;
}

std::string LlamaCppVlm::token_piece(llama_token token) const {
    std::vector<char> buffer(64);
    int32_t length = llama_token_to_piece(
        vocab_, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, false);
    if (length < 0 && length != std::numeric_limits<int32_t>::min()) {
        buffer.resize(static_cast<size_t>(-length));
        length = llama_token_to_piece(
            vocab_, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, false);
    }
    if (length < 0) return {};
    return std::string(buffer.data(), static_cast<size_t>(length));
}

SamplerPtr LlamaCppVlm::make_sampler(
    const unirt_SamplerConfig* config, int32_t& error) const {
    error = UNIRT_SUCCESS;
    if (config && !valid_sampler_config(*config)) {
        error = UNIRT_ERROR_COMMON_INVALID_INPUT;
        return {};
    }

    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    SamplerPtr chain(llama_sampler_chain_init(chain_params));
    if (!chain) {
        error = UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
        return {};
    }

    std::string grammar;
    if (config) {
        const int grammar_sources =
            (config->grammar_string && config->grammar_string[0] ? 1 : 0) +
            (config->grammar_path && config->grammar_path[0] ? 1 : 0) +
            (config->enable_json ? 1 : 0) +
            (config->json_schema && config->json_schema[0] ? 1 : 0);
        if (grammar_sources > 1) {
            error = UNIRT_ERROR_COMMON_INVALID_INPUT;
            return {};
        }
        if (config->grammar_string && config->grammar_string[0]) {
            grammar = config->grammar_string;
        } else if (config->grammar_path && config->grammar_path[0]) {
            if (!read_text_file(config->grammar_path, grammar)) {
                error = UNIRT_ERROR_COMMON_FILE_NOT_FOUND;
                return {};
            }
        } else if (config->enable_json) {
            grammar = kJsonGrammar;
        } else if (config->json_schema && config->json_schema[0]) {
            grammar = schema_to_gbnf(config->json_schema, error);
            if (grammar.empty()) return {};
        }
    }
    if (!grammar.empty()) {
        auto* grammar_sampler = llama_sampler_init_grammar(vocab_, grammar.c_str(), "root");
        if (!grammar_sampler) {
            error = UNIRT_ERROR_COMMON_INVALID_INPUT;
            return {};
        }
        llama_sampler_chain_add(chain.get(), grammar_sampler);
    }

    const float temperature = config ? config->temperature : 0.0f;
    if (config &&
        (config->repetition_penalty > 0.0f || config->presence_penalty != 0.0f ||
         config->frequency_penalty != 0.0f)) {
        llama_sampler_chain_add(
            chain.get(), llama_sampler_init_penalties(
                             64,
                             config->repetition_penalty > 0.0f
                                 ? config->repetition_penalty
                                 : 1.0f,
                             config->frequency_penalty, config->presence_penalty));
    }
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        return chain;
    }
    if (config->top_k > 0) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(config->top_k));
    }
    if (config->top_p > 0.0f && config->top_p < 1.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(config->top_p, 1));
    }
    if (config->min_p > 0.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(config->min_p, 1));
    }
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(temperature));
    const uint32_t seed = config->seed > 0
                              ? static_cast<uint32_t>(config->seed)
                              : LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(seed));
    return chain;
}

int32_t LlamaCppVlm::decode_token(llama_token token, llama_pos position) {
    BatchGuard guard;
    auto& batch = guard.batch;
    if (!batch.token || !batch.pos || !batch.seq_id || !batch.seq_id[0] ||
        !batch.n_seq_id || !batch.logits) {
        return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
    }
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
    const int32_t result = llama_decode(context_.get(), batch);
    if (result == 0) return UNIRT_SUCCESS;
    if (result == 1) return UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    if (result == 2) return UNIRT_ERROR_COMMON_CANCELLED;
    return UNIRT_ERROR_VLM_GENERATION_FAILED;
}

int32_t LlamaCppVlm::generate(
    const unirt_VlmGenerateInput* input,
    unirt_VlmGenerateOutput* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!context_ || !multimodal_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !input->prompt_utf8 || !output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *output = {};

    int32_t max_tokens = 512;
    int32_t requested_past = 0;
    const unirt_SamplerConfig* sampler_config = nullptr;
    std::vector<std::string> stops;
    const unirt_GenerationConfig* config = input->config;
    if (config) {
        if (config->stop_count < 0 || config->image_count < 0 ||
            config->audio_count < 0 || config->n_past < 0 ||
            (config->stop_count > 0 && !config->stop) ||
            (config->image_count > 0 && !config->image_paths) ||
            (config->audio_count > 0 && !config->audio_paths)) {
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        if (config->max_tokens > 0) max_tokens = config->max_tokens;
        requested_past = config->n_past;
        sampler_config = config->sampler_config;
        if (config->sliding_window || config->sliding_window_n_keep != 0 ||
            config->image_max_length != 0) {
            return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
        }
        for (int32_t index = 0; index < config->stop_count; ++index) {
            if (!config->stop[index]) return UNIRT_ERROR_COMMON_INVALID_INPUT;
            if (config->stop[index] && config->stop[index][0]) {
                stops.emplace_back(config->stop[index]);
            }
        }
        for (int32_t index = 0; index < config->image_count; ++index) {
            if (!config->image_paths[index] || !config->image_paths[index][0]) {
                return UNIRT_ERROR_COMMON_INVALID_INPUT;
            }
        }
        for (int32_t index = 0; index < config->audio_count; ++index) {
            if (!config->audio_paths[index] || !config->audio_paths[index][0]) {
                return UNIRT_ERROR_COMMON_INVALID_INPUT;
            }
        }
    }
    if (requested_past != 0) {
        UNIRT_LOG_ERROR("llama_cpp VLM: n_past>0 is not supported for media position state");
        return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }

    llama_memory_clear(llama_get_memory(context_.get()), true);
    n_past_ = 0;

    const int32_t image_count = config ? config->image_count : 0;
    const int32_t audio_count = config ? config->audio_count : 0;
    if (image_count > 0 && !mtmd_support_vision(multimodal_.get())) {
        return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    if (audio_count > 0 && !mtmd_support_audio(multimodal_.get())) {
        return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }

    std::string prompt = input->prompt_utf8;
    const char* marker_ptr = mtmd_get_marker(multimodal_.get());
    const std::string marker = marker_ptr && marker_ptr[0]
                                   ? marker_ptr
                                   : mtmd_default_marker();
    const size_t media_count = static_cast<size_t>(image_count) +
                               static_cast<size_t>(audio_count);
    size_t marker_count = count_markers(prompt, marker);
    if (media_count > 0 && marker_count == 0) {
        std::string prefix;
        for (size_t index = 0; index < media_count; ++index) prefix += marker;
        prompt.insert(0, prefix);
        marker_count = media_count;
    }
    if (marker_count != media_count) {
        UNIRT_LOG_ERROR(
            "llama_cpp VLM: prompt has {} media markers but {} files were supplied",
            marker_count, media_count);
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }

    std::vector<MediaKind> media_order;
    size_t tagged_markers = 0;
    size_t marker_offset = 0;
    while ((marker_offset = prompt.find(marker, marker_offset)) != std::string::npos) {
        const size_t tag_offset = marker_offset + marker.size();
        if (prompt.compare(tag_offset, kImageMediaTag.size(), kImageMediaTag) == 0) {
            prompt.erase(tag_offset, kImageMediaTag.size());
            media_order.push_back(MediaKind::image);
            ++tagged_markers;
        } else if (prompt.compare(tag_offset, kAudioMediaTag.size(), kAudioMediaTag) == 0) {
            prompt.erase(tag_offset, kAudioMediaTag.size());
            media_order.push_back(MediaKind::audio);
            ++tagged_markers;
        }
        marker_offset = tag_offset;
    }
    if (tagged_markers != 0 && tagged_markers != marker_count) {
        UNIRT_LOG_ERROR("llama_cpp VLM: prompt mixes typed and untyped media markers");
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    if (tagged_markers == 0) {
        // A direct generate() call can use generic markers, but the split C ABI
        // can only express the deterministic images-then-audios ordering.
        media_order.insert(media_order.end(), static_cast<size_t>(image_count), MediaKind::image);
        media_order.insert(media_order.end(), static_cast<size_t>(audio_count), MediaKind::audio);
    }
    const size_t expected_images = static_cast<size_t>(std::count(
        media_order.begin(), media_order.end(), MediaKind::image));
    const size_t expected_audios = media_order.size() - expected_images;
    if (expected_images != static_cast<size_t>(image_count) ||
        expected_audios != static_cast<size_t>(audio_count)) {
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }

    mtmd::bitmaps bitmaps;
    bitmaps.entries.reserve(media_count);
    size_t image_index = 0;
    size_t audio_index = 0;
    for (MediaKind kind : media_order) {
        const char* path = nullptr;
        if (kind == MediaKind::image) {
            path = config->image_paths[image_index++];
        } else {
            path = config->audio_paths[audio_index++];
        }
        if (!readable_file(path)) {
            return kind == MediaKind::image
                       ? UNIRT_ERROR_VLM_IMAGE_LOAD
                       : UNIRT_ERROR_VLM_AUDIO_LOAD;
        }
        auto loaded = mtmd_helper_bitmap_init_from_file(multimodal_.get(), path, false);
        if (!loaded.bitmap) {
            if (loaded.video_ctx) mtmd_helper_video_free(loaded.video_ctx);
            return kind == MediaKind::image
                       ? UNIRT_ERROR_VLM_IMAGE_FORMAT
                       : UNIRT_ERROR_VLM_AUDIO_FORMAT;
        }
        if (loaded.video_ctx) {
            // The public UniRT ABI currently advertises image/audio, not video.
            mtmd_bitmap_free(loaded.bitmap);
            mtmd_helper_video_free(loaded.video_ctx);
            return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
        }
        const bool is_audio = mtmd_bitmap_is_audio(loaded.bitmap);
        if ((kind == MediaKind::audio) != is_audio) {
            mtmd_bitmap_free(loaded.bitmap);
            return kind == MediaKind::image
                       ? UNIRT_ERROR_VLM_IMAGE_FORMAT
                       : UNIRT_ERROR_VLM_AUDIO_FORMAT;
        }
        bitmaps.entries.emplace_back(loaded.bitmap);
    }

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (!chunks.ptr) return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
    mtmd_input_text text{};
    text.text = prompt.data();
    text.text_len = prompt.size();
    text.add_special = true;
    text.parse_special = true;
    auto bitmap_ptrs = bitmaps.c_ptr();
    const int32_t tokenize_result = mtmd_tokenize(
        multimodal_.get(), chunks.ptr.get(), &text,
        bitmap_ptrs.empty() ? nullptr : bitmap_ptrs.data(), bitmap_ptrs.size());
    if (tokenize_result == 1) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    if (tokenize_result != 0) {
        return image_count > 0
                   ? UNIRT_ERROR_VLM_IMAGE_FORMAT
                   : UNIRT_ERROR_VLM_AUDIO_FORMAT;
    }

    const size_t prompt_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
    const llama_pos prompt_positions = mtmd_helper_get_n_pos(chunks.ptr.get());
    if (prompt_tokens == 0 || prompt_positions < 0 ||
        prompt_positions > static_cast<llama_pos>(context_size_)) {
        return UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    }

    int32_t sampler_error = UNIRT_SUCCESS;
    SamplerPtr sampler = make_sampler(sampler_config, sampler_error);
    if (!sampler) return sampler_error;

    const int64_t start_time = monotonic_us();
    llama_pos new_n_past = 0;
    const int32_t eval_result = mtmd_helper_eval_chunks(
        multimodal_.get(), context_.get(), chunks.ptr.get(), 0, 0,
        batch_size_, true, &new_n_past);
    if (eval_result != 0 || new_n_past < 0) {
        llama_memory_clear(llama_get_memory(context_.get()), true);
        n_past_ = 0;
        return eval_result == 1
                   ? UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH
                   : UNIRT_ERROR_VLM_GENERATION_FAILED;
    }
    n_past_ = new_n_past;

    StopStreamState stream_state(std::move(stops));
    const char* stop_reason = "length";
    int32_t generated = 0;
    int32_t result = UNIRT_SUCCESS;
    int64_t first_token_time = monotonic_us();

    while (generated < max_tokens) {
        if (n_past_ >= static_cast<llama_pos>(context_size_)) {
            stop_reason = "context_length";
            if (!stream_state.emit_safe(input->on_token, input->user_data, true)) {
                stream_state.discard_unemitted();
                stop_reason = "user";
            }
            break;
        }
        const llama_token token = llama_sampler_sample(sampler.get(), context_.get(), -1);
        if (token == LLAMA_TOKEN_NULL) {
            result = UNIRT_ERROR_VLM_GENERATION_FAILED;
            break;
        }
        if (llama_vocab_is_eog(vocab_, token)) {
            stop_reason = "eos";
            if (!stream_state.emit_safe(input->on_token, input->user_data, true)) {
                stream_state.discard_unemitted();
                stop_reason = "user";
            }
            break;
        }
        if (generated == 0) first_token_time = monotonic_us();
        const std::string piece = token_piece(token);
        result = decode_token(token, n_past_);
        if (result != UNIRT_SUCCESS) break;
        ++n_past_;

        stream_state.append(piece);
        ++generated;
        const bool hit_stop = stream_state.find_and_trim_stop();
        if (!stream_state.emit_safe(input->on_token, input->user_data, hit_stop)) {
            stream_state.discard_unemitted();
            stop_reason = "user";
            break;
        }
        if (hit_stop) {
            stop_reason = "stop_sequence";
            break;
        }
        if (generated >= max_tokens) {
            if (!stream_state.emit_safe(input->on_token, input->user_data, true)) {
                stream_state.discard_unemitted();
                stop_reason = "user";
            }
            break;
        }
    }

    const int64_t end_time = monotonic_us();
    if (result != UNIRT_SUCCESS) return result;
    stream_state.discard_incomplete_utf8_tail();
    output->full_text = copy_to_c(stream_state.text());
    if (!output->full_text) return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;

    auto& profile = output->profile_data;
    profile = {};
    profile.ttft = first_token_time - start_time;
    profile.prompt_time = first_token_time - start_time;
    profile.decode_time = end_time - first_token_time;
    profile.prompt_tokens = static_cast<int64_t>(prompt_tokens);
    profile.generated_tokens = generated;
    profile.prefill_speed = profile.prompt_time > 0
                                ? profile.prompt_tokens * 1000000.0 / profile.prompt_time
                                : 0.0;
    profile.decoding_speed = profile.decode_time > 0
                                 ? profile.generated_tokens * 1000000.0 / profile.decode_time
                                 : 0.0;
    profile.stop_reason = stop_reason;
    return UNIRT_SUCCESS;
}

}  // namespace unirt::llama_plugin

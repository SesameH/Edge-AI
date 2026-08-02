// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>

#include "logging.h"
#include "generation_state.h"
#include "device_label.h"
#include "schema_grammar.h"
#include "weight_cache.h"

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

// Reports the model's own log-probabilities for one decode step.
//
// Deliberately reads llama_get_logits_ith() rather than anything the sampler
// chain produced: those are the raw model scores, before temperature, top_p or
// a grammar has touched the distribution. That is what makes the number usable
// as a confidence signal -- it says what the model believed, not what the
// sampling settings allowed. Skipped entirely when logprobs is 0, since it
// costs a full softmax over the vocabulary per token.
class LogprobReporter {
   public:
    LogprobReporter(const llama_vocab* vocab, int32_t alternatives)
        : vocab_(vocab), alternatives_(alternatives) {}

    /** Emit the sampled token and its top alternatives. False = caller stopped us. */
    bool report(
        llama_context* context, llama_token sampled, unirt_logprob_callback callback,
        void* user_data) {
        const float* logits = llama_get_logits_ith(context, -1);
        const int32_t vocab_size = llama_vocab_n_tokens(vocab_);
        if (!logits || vocab_size <= 0) return true;

        // log-softmax the stable way: subtracting the max keeps exp() in range
        // for the large-magnitude logits that quantized models produce.
        float max_logit = logits[0];
        for (int32_t id = 1; id < vocab_size; ++id) max_logit = std::max(max_logit, logits[id]);
        double sum = 0.0;
        for (int32_t id = 0; id < vocab_size; ++id) sum += std::exp(logits[id] - max_logit);
        const double log_normalizer = std::log(sum) + max_logit;

        ranked_.resize(static_cast<size_t>(vocab_size));
        for (int32_t id = 0; id < vocab_size; ++id) ranked_[static_cast<size_t>(id)] = id;
        // The sampled token comes first whether or not it is the most likely
        // one -- with any temperature above zero it often is not.
        const size_t wanted = std::min<size_t>(
            static_cast<size_t>(alternatives_), ranked_.size());
        if (wanted > 0) {
            std::partial_sort(
                ranked_.begin(), ranked_.begin() + static_cast<ptrdiff_t>(wanted), ranked_.end(),
                [logits](llama_token a, llama_token b) { return logits[a] > logits[b]; });
        }

        entries_.clear();
        pieces_.clear();
        pieces_.reserve(wanted + 1);
        append(sampled, logits, log_normalizer);
        for (size_t index = 0; index < wanted; ++index) {
            if (ranked_[index] == sampled) continue;
            append(ranked_[index], logits, log_normalizer);
        }
        // The piece strings are only stable once the vector has stopped
        // growing, so the pointers go in last.
        for (size_t index = 0; index < entries_.size(); ++index) {
            entries_[index].piece = pieces_[index].c_str();
        }
        return callback(entries_.data(), static_cast<int32_t>(entries_.size()), user_data);
    }

   private:
    void append(llama_token token, const float* logits, double log_normalizer) {
        pieces_.push_back(piece_of(token));
        unirt_Logprob entry{};
        entry.token_id = token;
        entry.logprob  = static_cast<float>(logits[token] - log_normalizer);
        entries_.push_back(entry);
    }

    std::string piece_of(llama_token token) const {
        std::vector<char> buffer(64);
        int32_t length = llama_token_to_piece(
            vocab_, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
        if (length < 0 && length != std::numeric_limits<int32_t>::min()) {
            buffer.resize(static_cast<size_t>(-length));
            length = llama_token_to_piece(
                vocab_, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
        }
        if (length < 0) return {};
        return std::string(buffer.data(), static_cast<size_t>(length));
    }

    const llama_vocab*         vocab_;
    int32_t                    alternatives_;
    std::vector<llama_token>   ranked_;
    std::vector<std::string>   pieces_;
    std::vector<unirt_Logprob> entries_;
};

bool valid_sampler_config(const unirt_SamplerConfig& config) {
    return config.top_k >= 0 &&
           std::isfinite(config.temperature) && config.temperature >= 0.0f &&
           std::isfinite(config.top_p) && config.top_p >= 0.0f && config.top_p <= 1.0f &&
           std::isfinite(config.min_p) && config.min_p >= 0.0f && config.min_p <= 1.0f &&
           std::isfinite(config.repetition_penalty) && config.repetition_penalty >= 0.0f &&
           std::isfinite(config.presence_penalty) &&
           std::isfinite(config.frequency_penalty);
}

}  // namespace

void ModelDeleter::operator()(llama_model* model) const noexcept {
    if (model) llama_model_free(model);
}

void ContextDeleter::operator()(llama_context* context) const noexcept {
    if (context) llama_free(context);
}

void SamplerDeleter::operator()(llama_sampler* sampler) const noexcept {
    if (sampler) llama_sampler_free(sampler);
}

void LlamaCppLlm::clear_model() noexcept {
    context_.reset();
    gguf_model_.reset();
    vocab_ = nullptr;
    devices_.clear();
    history_.clear();
    template_override_.clear();
    device_name_ = "CPU";
    context_size_ = 0;
}

int32_t LlamaCppLlm::create(const unirt_LlmCreateInput* input) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!input || !input->model_path || !input->model_path[0]) {
        UNIRT_LOG_ERROR("llama_cpp: create requires a non-empty model_path");
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }

    std::string requested_template;
    if (input->config.chat_template_content && input->config.chat_template_content[0]) {
        requested_template = input->config.chat_template_content;
    } else if (input->config.chat_template_path && input->config.chat_template_path[0] &&
               !read_text_file(input->config.chat_template_path, requested_template)) {
        UNIRT_LOG_ERROR(
            "llama_cpp: cannot read chat template {}", input->config.chat_template_path);
        return UNIRT_ERROR_COMMON_FILE_NOT_FOUND;
    }

    std::vector<ggml_backend_dev_t> selected_devices;
    std::string                     selected_device_name = "CPU";
    llama_model_params              model_params = llama_model_default_params();
    model_params.n_gpu_layers = input->config.n_gpu_layers;

    if (input->device_id && input->device_id[0]) {
        ggml_backend_dev_t device = ggml_backend_dev_by_name(input->device_id);
        if (!device) {
            UNIRT_LOG_ERROR("llama_cpp: device '{}' does not exist", input->device_id);
            return UNIRT_ERROR_COMMON_INVALID_DEVICE;
        }
        selected_devices = {device, nullptr};
        model_params.devices = selected_devices.data();
        selected_device_name = device_label(device, input->device_id);
    } else if (model_params.n_gpu_layers == 0) {
        if (auto* device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            selected_devices = {device, nullptr};
            model_params.devices = selected_devices.data();
            selected_device_name = device_label(device, "CPU");
        }
    } else {
        if (auto* device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
            selected_device_name = device_label(device, "GPU");
        }
    }

    // device_key is what stops two handles sharing weights they cannot share:
    // an explicit device pin bakes a different placement into the load.
    const std::string device_key =
        input->device_id && input->device_id[0] ? input->device_id : selected_device_name;
    SharedModel model = acquire_model(input->model_path, device_key, model_params);
    if (!model) {
        UNIRT_LOG_ERROR("llama_cpp: failed to load GGUF model {}", input->model_path);
        return UNIRT_ERROR_COMMON_MODEL_LOAD;
    }
    if (llama_model_has_encoder(model.get())) {
        UNIRT_LOG_ERROR("llama_cpp: encoder-decoder models are not supported by the LlmBackend API");
        return UNIRT_ERROR_COMMON_MODEL_INVALID;
    }

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
    if (model_params.n_gpu_layers == 0) {
        context_params.offload_kqv = false;
        context_params.op_offload = false;
    }

    ContextPtr context(llama_init_from_model(model.get(), context_params));
    if (!context && (!input->device_id || !input->device_id[0]) &&
        model_params.n_gpu_layers != 0) {
        // A compiled GPU backend may be enumerated but unavailable at runtime
        // (notably Metal in headless macOS sessions). Preserve the useful
        // default-offload behavior while keeping local inference available.
        UNIRT_LOG_WARN("llama_cpp: accelerator initialization failed; retrying on CPU");
        model.reset();
        model_params.n_gpu_layers = 0;
        if (auto* cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            selected_devices = {cpu, nullptr};
            model_params.devices = selected_devices.data();
            selected_device_name = device_label(cpu, "CPU");
        } else {
            model_params.devices = nullptr;
        }
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        model = acquire_model(input->model_path, selected_device_name, model_params);
        if (model) context.reset(llama_init_from_model(model.get(), context_params));
    }
    if (!context) {
        UNIRT_LOG_ERROR("llama_cpp: failed to allocate inference context");
        return UNIRT_ERROR_COMMON_MODEL_LOAD;
    }

    const int32_t actual_context = static_cast<int32_t>(llama_n_ctx(context.get()));
    if (actual_context <= 0 || llama_n_batch(context.get()) == 0) {
        UNIRT_LOG_ERROR("llama_cpp: backend returned an invalid context configuration");
        return UNIRT_ERROR_COMMON_MODEL_LOAD;
    }

    clear_model();
    gguf_model_ = std::move(model);
    context_ = std::move(context);
    vocab_ = llama_model_get_vocab(gguf_model_.get());
    devices_ = std::move(selected_devices);
    template_override_ = std::move(requested_template);
    device_name_ = std::move(selected_device_name);
    context_size_ = actual_context;
    shift_supported_ = llama_memory_can_shift(llama_get_memory(context_.get())) &&
                       !llama_model_is_recurrent(gguf_model_.get());

    char description[256] = {};
    llama_model_desc(gguf_model_.get(), description, sizeof(description));
    UNIRT_LOG_INFO(
        "llama_cpp: loaded {} [{}], context={}, batch={}, device={}", input->model_path,
        description[0] ? description : "GGUF", context_size_, llama_n_batch(context_.get()),
        device_name_);
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!context_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    llama_memory_clear(llama_get_memory(context_.get()), true);
    history_.clear();
    llama_perf_context_reset(context_.get());
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::save_kv_cache(
    const unirt_KvCacheSaveInput* input, unirt_KvCacheSaveOutput*) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!context_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !input->path || !input->path[0]) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    if (!llama_state_save_file(
            context_.get(), input->path, history_.data(), history_.size())) {
        UNIRT_LOG_ERROR("llama_cpp: failed to save session {}", input->path);
        return UNIRT_ERROR_COMMON_UNKNOWN;
    }
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::load_kv_cache(
    const unirt_KvCacheLoadInput* input, unirt_KvCacheLoadOutput*) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!context_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !input->path || !input->path[0]) return UNIRT_ERROR_COMMON_INVALID_INPUT;

    std::vector<llama_token> loaded(static_cast<size_t>(context_size_));
    size_t                   count = 0;
    if (!llama_state_load_file(
            context_.get(), input->path, loaded.data(), loaded.size(), &count)) {
        UNIRT_LOG_ERROR("llama_cpp: failed to load session {}", input->path);
        return UNIRT_ERROR_COMMON_MODEL_INVALID;
    }
    if (count > loaded.size()) {
        UNIRT_LOG_ERROR("llama_cpp: session contains too many tokens ({})", count);
        return UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    }
    loaded.resize(count);
    history_ = std::move(loaded);
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::apply_chat_template(
    const unirt_LlmApplyChatTemplateInput* input,
    unirt_LlmApplyChatTemplateOutput* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!gguf_model_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output || input->message_count < 0 ||
        (input->message_count > 0 && !input->messages)) {
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    output->formatted_text = nullptr;

    if (input->tools && input->tools[0]) {
        UNIRT_LOG_DEBUG(
            "llama_cpp: tools were supplied, but llama_chat_apply_template has no tools parameter");
    }

    std::vector<llama_chat_message> messages;
    messages.reserve(static_cast<size_t>(input->message_count));
    size_t estimated_size = 256;
    for (int32_t index = 0; index < input->message_count; ++index) {
        const auto& source = input->messages[index];
        const char* role = source.role ? source.role : "user";
        const char* content = source.content ? source.content : "";
        messages.push_back({role, content});
        estimated_size += std::strlen(role) + std::strlen(content) + 32;
    }
    estimated_size = std::min<size_t>(
        std::max<size_t>(estimated_size * 2, 512),
        static_cast<size_t>(std::numeric_limits<int32_t>::max()));

    const char* tmpl = template_override_.empty()
                           ? llama_model_chat_template(gguf_model_.get(), nullptr)
                           : template_override_.c_str();
    std::vector<char> buffer(estimated_size);
    int32_t length = llama_chat_apply_template(
        tmpl, messages.data(), messages.size(), input->add_generation_prompt, buffer.data(),
        static_cast<int32_t>(buffer.size()));
    if (length > static_cast<int32_t>(buffer.size())) {
        buffer.resize(static_cast<size_t>(length));
        length = llama_chat_apply_template(
            tmpl, messages.data(), messages.size(), input->add_generation_prompt, buffer.data(),
            static_cast<int32_t>(buffer.size()));
    }
    if (length < 0 || length > static_cast<int32_t>(buffer.size())) {
        UNIRT_LOG_ERROR("llama_cpp: model chat template is not supported by llama.cpp");
        return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }

    output->formatted_text = copy_to_c(std::string(buffer.data(), static_cast<size_t>(length)));
    return output->formatted_text ? UNIRT_SUCCESS : UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
}

int32_t LlamaCppLlm::tokenize(
    const char* text, bool add_special, std::vector<llama_token>& output) const {
    if (!text) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    const size_t length = std::strlen(text);
    if (length > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return UNIRT_ERROR_LLM_TOKENIZATION_FAILED;
    }

    size_t capacity = length + (add_special ? 2 : 0);
    capacity = std::min<size_t>(
        capacity, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    output.resize(capacity);
    int32_t count = llama_tokenize(
        vocab_, text, static_cast<int32_t>(length), output.data(),
        static_cast<int32_t>(output.size()), add_special, true);
    if (count == std::numeric_limits<int32_t>::min()) {
        output.clear();
        return UNIRT_ERROR_LLM_TOKENIZATION_FAILED;
    }
    if (count < 0) {
        output.resize(static_cast<size_t>(-count));
        count = llama_tokenize(
            vocab_, text, static_cast<int32_t>(length), output.data(),
            static_cast<int32_t>(output.size()), add_special, true);
    }
    if (count < 0) {
        output.clear();
        return UNIRT_ERROR_LLM_TOKENIZATION_FAILED;
    }
    output.resize(static_cast<size_t>(count));
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::evict_for_space(int32_t incoming) {
    if (!shift_supported_) return 0;
    const int32_t cached = static_cast<int32_t>(history_.size());
    const int32_t head   = std::min(pinned_head_, cached);
    // Free at least enough for the incoming tokens, but take half of the
    // evictable region so we do not shift on every subsequent token.
    const int32_t deficit  = cached + incoming - context_size_ + 1;
    int32_t       n_evict  = std::max((cached - head) / 2, deficit);
    n_evict                = std::min(n_evict, cached - head);
    if (n_evict <= 0) return 0;

    auto* memory = llama_get_memory(context_.get());
    llama_memory_seq_rm(memory, 0, head, head + n_evict);
    llama_memory_seq_add(memory, 0, head + n_evict, cached, -n_evict);
    history_.erase(history_.begin() + head, history_.begin() + head + n_evict);
    UNIRT_LOG_INFO(
        "llama_cpp: context shift evicted {} tokens (head {}, cached {} -> {})", n_evict, head,
        cached, history_.size());
    return n_evict;
}

int32_t LlamaCppLlm::decode(const llama_token* tokens, int32_t count) {
    if (count <= 0) return UNIRT_SUCCESS;
    const int32_t batch_size = static_cast<int32_t>(llama_n_batch(context_.get()));
    for (int32_t offset = 0; offset < count; offset += batch_size) {
        const int32_t chunk_size = std::min(batch_size, count - offset);
        auto* writable = const_cast<llama_token*>(tokens + offset);
        int32_t result = llama_decode(
            context_.get(), llama_batch_get_one(writable, chunk_size));
        // rc==1 means the batch found no KV slot; shift the window and retry
        // instead of failing, as long as eviction keeps making progress.
        while (result == 1 && evict_for_space(chunk_size) > 0) {
            result = llama_decode(
                context_.get(), llama_batch_get_one(writable, chunk_size));
        }
        if (result == 0) {
            // history_ mirrors the KV contents at all times so eviction and
            // prefix reuse can reason about positions from it alone.
            history_.insert(history_.end(), tokens + offset, tokens + offset + chunk_size);
            continue;
        }
        UNIRT_LOG_ERROR("llama_cpp: decode failed with code {}", result);
        if (result == 1) return UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
        if (result == 2) return UNIRT_ERROR_COMMON_CANCELLED;
        return UNIRT_ERROR_LLM_GENERATION_FAILED;
    }
    return UNIRT_SUCCESS;
}

size_t LlamaCppLlm::reusable_prefix(const std::vector<llama_token>& wanted) const {
    if (history_.empty() || wanted.empty()) return 0;
    const size_t limit = std::min(history_.size(), wanted.size() - 1);
    size_t       shared = 0;
    while (shared < limit && history_[shared] == wanted[shared]) ++shared;
    return shared;
}

std::string LlamaCppLlm::token_piece(llama_token token) const {
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

SamplerPtr LlamaCppLlm::make_sampler(
    const unirt_SamplerConfig* config, int32_t& error, bool& has_grammar) const {
    error       = UNIRT_SUCCESS;
    has_grammar = false;
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
            UNIRT_LOG_ERROR("llama_cpp: invalid GBNF grammar");
            error = UNIRT_ERROR_COMMON_INVALID_INPUT;
            return {};
        }
        llama_sampler_chain_add(chain.get(), grammar_sampler);
        has_grammar = true;
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

int32_t LlamaCppLlm::generate(
    const unirt_LlmGenerateInput* input, unirt_LlmGenerateOutput* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!context_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *output = {};

    if (input->input_ids_count < 0 ||
        (input->input_ids_count > 0 && !input->input_ids)) {
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }

    int32_t                  max_tokens = 512;
    std::vector<std::string> stops;
    const unirt_SamplerConfig* sampler_config = nullptr;
    int32_t requested_past = 0;
    int32_t logprob_alternatives = 0;
    if (input->config) {
        if (input->config->max_tokens > 0) max_tokens = input->config->max_tokens;
        if (input->config->stop_count < 0 ||
            (input->config->stop_count > 0 && !input->config->stop)) {
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        for (int32_t index = 0; index < input->config->stop_count; ++index) {
            if (input->config->stop[index] && input->config->stop[index][0]) {
                stops.emplace_back(input->config->stop[index]);
            }
        }
        sampler_config = input->config->sampler_config;
        requested_past = input->config->n_past;
        if (input->config->logprobs < 0) return UNIRT_ERROR_COMMON_INVALID_INPUT;
        logprob_alternatives = input->config->logprobs;
        if (input->config->sliding_window_n_keep > 0) {
            pinned_head_ = input->config->sliding_window_n_keep;
        }
    }

    if (requested_past < 0 || static_cast<size_t>(requested_past) > history_.size()) {
        UNIRT_LOG_ERROR(
            "llama_cpp: n_past={} is invalid for {} cached tokens", requested_past,
            history_.size());
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    // n_past > 0 is an explicit rewind to that point; n_past == 0 leaves the
    // cache alone and lets the prefix-reuse pass below decide what survives.
    if (requested_past > 0 && static_cast<size_t>(requested_past) < history_.size()) {
        if (!llama_memory_seq_rm(
                llama_get_memory(context_.get()), 0, requested_past, -1)) {
            UNIRT_LOG_ERROR("llama_cpp: backend cannot trim KV cache to n_past={}", requested_past);
            return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
        }
        history_.resize(static_cast<size_t>(requested_past));
    }

    std::vector<llama_token> prompt;
    if (input->input_ids && input->input_ids_count > 0) {
        prompt.assign(input->input_ids, input->input_ids + input->input_ids_count);
    } else if (input->prompt_utf8) {
        // BOS belongs to the sequence, not to the call, so the decision is
        // n_past's and not history_'s. With n_past == 0 the prompt is the whole
        // prompt -- a chat client resends the entire transcript -- so it starts
        // at BOS every time; keying this off "is the cache empty" instead
        // dropped BOS from turn two onwards on every add_bos model (Gemma,
        // Llama, Mistral). That is wrong twice over: the model loses a token it
        // was trained to always see, and because history_[0] is then the cached
        // BOS while prompt[0] is the first real token, the prefix match fails
        // at position 0 and the whole transcript is re-prefilled each turn.
        // n_past > 0 is an explicit continuation from a caller managing its own
        // positions, and mid-sequence is exactly where BOS must not appear.
        const int32_t tokenize_result = tokenize(input->prompt_utf8, requested_past == 0, prompt);
        if (tokenize_result != UNIRT_SUCCESS) return tokenize_result;
    } else {
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    if (prompt.empty()) return UNIRT_ERROR_COMMON_INVALID_INPUT;

    // Skip re-evaluating whatever prefix of the prompt is already in the KV
    // cache (the common chat pattern resends the whole transcript each turn).
    // Anything cached beyond the shared prefix is stale and gets dropped.
    const size_t shared = reusable_prefix(prompt);
    if (shared < history_.size()) {
        auto* memory = llama_get_memory(context_.get());
        if (shared == 0) {
            llama_memory_clear(memory, true);
        } else if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(shared), -1)) {
            llama_memory_clear(memory, true);
            history_.clear();
        }
        history_.resize(std::min(history_.size(), shared));
    }
    const auto* fresh_tokens = prompt.data() + history_.size();
    const auto  fresh_count  = static_cast<int32_t>(prompt.size() - history_.size());
    if (history_.size() > 0) {
        UNIRT_LOG_DEBUG(
            "llama_cpp: prefix cache reused {} tokens, evaluating {}", history_.size(), fresh_count);
    }

    const size_t used = history_.size();
    if (!shift_supported_ &&
        (used > static_cast<size_t>(context_size_) ||
         prompt.size() > static_cast<size_t>(context_size_))) {
        UNIRT_LOG_ERROR(
            "llama_cpp: prompt would exceed context (used={}, prompt={}, capacity={})", used,
            prompt.size(), context_size_);
        return UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    }

    int32_t sampler_error = UNIRT_SUCCESS;
    bool    sampler_has_grammar = false;
    SamplerPtr sampler = make_sampler(sampler_config, sampler_error, sampler_has_grammar);
    if (!sampler) return sampler_error;

    // Give the penalty samplers their context window. A grammar sampler in
    // the chain must not see prompt tokens (they are not grammar output), so
    // skip in that case.
    if (!sampler_has_grammar) {
        const size_t window = std::min<size_t>(prompt.size(), 64);
        for (size_t i = prompt.size() - window; i < prompt.size(); ++i) {
            llama_sampler_accept(sampler.get(), prompt[i]);
        }
    }

    const int64_t start_time = monotonic_us();
    int32_t result = decode(fresh_tokens, fresh_count);
    if (result != UNIRT_SUCCESS) return result;

    StopStreamState stream_state(std::move(stops));
    const char* stop_reason = "length";
    int32_t generated = 0;
    int64_t first_token_time = monotonic_us();

    std::optional<LogprobReporter> reporter;
    if (input->on_logprob && logprob_alternatives >= 0 && input->config &&
        input->config->logprobs > 0) {
        reporter.emplace(vocab_, logprob_alternatives);
    }

    while (generated < max_tokens) {
        if (history_.size() >= static_cast<size_t>(context_size_) && evict_for_space(1) == 0) {
            stop_reason = "context_length";
            if (!stream_state.emit_safe(input->on_token, input->user_data, true)) {
                stream_state.discard_unemitted();
                stop_reason = "user";
            }
            break;
        }

        const llama_token token = llama_sampler_sample(sampler.get(), context_.get(), -1);
        if (token == LLAMA_TOKEN_NULL) {
            result = UNIRT_ERROR_LLM_GENERATION_FAILED;
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

        std::string piece = token_piece(token);
        if (generated == 0) first_token_time = monotonic_us();

        // Before decode(): the logits still belong to the step that produced
        // this token. Decoding it overwrites them with the next step's.
        if (reporter && !reporter->report(
                            context_.get(), token, input->on_logprob, input->user_data)) {
            stream_state.discard_unemitted();
            stop_reason = "user";
            break;
        }

        // Commit every non-EOG sampled token before exposing it.  This keeps
        // llama.cpp's KV state and history_ identical even for max-length,
        // callback cancellation, and stop-sequence exits.  decode() itself
        // appends the token to history_.
        result = decode(&token, 1);
        if (result != UNIRT_SUCCESS) break;

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
    profile.prompt_tokens = static_cast<int64_t>(prompt.size());
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

int32_t LlamaCppLlm::get_model_info(unirt_LlmModelInfo* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!gguf_model_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    output->vocab_size = llama_vocab_n_tokens(vocab_);
    const llama_token bos = llama_vocab_bos(vocab_);
    output->bos_token = bos == LLAMA_TOKEN_NULL ? -1 : bos;
    output->add_bos = llama_vocab_get_add_bos(vocab_) ? 1 : 0;
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::get_runtime_stats(unirt_LlmRuntimeStats* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!gguf_model_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    output->model_bytes = static_cast<int64_t>(llama_model_size(gguf_model_.get()));
    output->kv_cache_bytes = context_
                                 ? static_cast<int64_t>(llama_state_get_size(context_.get()))
                                 : -1;
    output->device_peak_bytes = -1;
    output->device_name = device_name_.c_str();
    return UNIRT_SUCCESS;
}

}  // namespace unirt::llama_plugin

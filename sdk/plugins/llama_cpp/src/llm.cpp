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
        const float* logits, llama_token sampled, unirt_logprob_callback callback,
        void* user_data) {
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

LlamaCppLlm::~LlamaCppLlm() { clear_model(); }

void LlamaCppLlm::clear_model() noexcept {
    if (engine_ && sequence_ != BatchEngine::kNoSequence) engine_->release(sequence_);
    sequence_ = BatchEngine::kNoSequence;
    engine_.reset();
    vocab_ = nullptr;
    last_logits_ = nullptr;
    devices_.clear();
    history_.clear();
    template_override_.clear();
    device_name_ = "CPU";
    context_size_ = 0;
}

int32_t DraftModel::load(
    const std::string& path, const llama_model_params& model_params,
    const llama_context_params& context_params, const std::string& device_key,
    const llama_vocab* target_vocab) {
    weights_ = acquire_model(path, device_key, model_params);
    if (!weights_) {
        UNIRT_LOG_ERROR("llama_cpp: failed to load draft model {}", path);
        return UNIRT_ERROR_COMMON_MODEL_LOAD;
    }
    vocab_ = llama_model_get_vocab(weights_.get());

    // Same vocabulary or nothing: a proposal is a token id, and against a
    // different vocabulary those ids mean different words. The check is the
    // token count plus a sample of the pieces -- two models can agree on size
    // and still disagree on content (a fine-tune with added special tokens).
    if (llama_vocab_n_tokens(vocab_) != llama_vocab_n_tokens(target_vocab)) {
        UNIRT_LOG_ERROR(
            "llama_cpp: draft model vocabulary has {} tokens against the target's {}; "
            "speculative decoding needs the same vocabulary",
            llama_vocab_n_tokens(vocab_), llama_vocab_n_tokens(target_vocab));
        weights_.reset();
        return UNIRT_ERROR_COMMON_MODEL_INVALID;
    }
    const int32_t count = llama_vocab_n_tokens(vocab_);
    for (int32_t probe = 0; probe < 64; ++probe) {
        const llama_token token = static_cast<llama_token>((int64_t)probe * count / 64);
        const char* mine   = llama_vocab_get_text(vocab_, token);
        const char* theirs = llama_vocab_get_text(target_vocab, token);
        if (!mine || !theirs || std::strcmp(mine, theirs) != 0) {
            UNIRT_LOG_ERROR(
                "llama_cpp: draft and target vocabularies differ at token {}; "
                "speculative decoding needs the same vocabulary", token);
            weights_.reset();
            return UNIRT_ERROR_COMMON_MODEL_INVALID;
        }
    }

    context_.reset(llama_init_from_model(weights_.get(), context_params));
    if (!context_) {
        UNIRT_LOG_ERROR("llama_cpp: failed to allocate a context for the draft model");
        weights_.reset();
        return UNIRT_ERROR_COMMON_MODEL_LOAD;
    }
    context_size_ = static_cast<int32_t>(llama_n_ctx(context_.get()));

    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    sampler_.reset(llama_sampler_chain_init(chain_params));
    if (!sampler_) {
        context_.reset();
        weights_.reset();
        return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
    }
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_greedy());

    char description[256] = {};
    llama_model_desc(weights_.get(), description, sizeof(description));
    UNIRT_LOG_INFO(
        "llama_cpp: draft model {} [{}] attached", path,
        description[0] ? description : "GGUF");
    return UNIRT_SUCCESS;
}

int32_t DraftModel::sync(const std::vector<llama_token>& prefix) {
    if (!context_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (static_cast<int32_t>(prefix.size()) >= context_size_) {
        // Out of window. Not an error: the caller drops back to plain decoding.
        return UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    }

    size_t shared = 0;
    while (shared < history_.size() && shared < prefix.size() &&
           history_[shared] == prefix[shared]) {
        ++shared;
    }
    if (shared < history_.size()) {
        auto* memory = llama_get_memory(context_.get());
        if (shared == 0) {
            llama_memory_clear(memory, true);
        } else if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(shared), -1)) {
            llama_memory_clear(memory, true);
            shared = 0;
        }
        history_.resize(shared);
    }
    if (history_.size() == prefix.size()) return UNIRT_SUCCESS;

    auto* tokens = const_cast<llama_token*>(prefix.data() + history_.size());
    const int32_t count = static_cast<int32_t>(prefix.size() - history_.size());
    const int32_t batch_size = static_cast<int32_t>(llama_n_batch(context_.get()));
    for (int32_t offset = 0; offset < count; offset += batch_size) {
        const int32_t chunk = std::min(batch_size, count - offset);
        if (llama_decode(context_.get(), llama_batch_get_one(tokens + offset, chunk)) != 0) {
            // Leave history_ describing what the KV really holds, or the next
            // sync would reuse a prefix that is not there.
            llama_memory_clear(llama_get_memory(context_.get()), true);
            history_.clear();
            return UNIRT_ERROR_LLM_GENERATION_FAILED;
        }
        history_.insert(history_.end(), tokens + offset, tokens + offset + chunk);
    }
    return UNIRT_SUCCESS;
}

std::vector<llama_token> DraftModel::propose(int32_t count) {
    std::vector<llama_token> proposals;
    if (!context_ || count <= 0) return proposals;
    proposals.reserve(static_cast<size_t>(count));
    for (int32_t index = 0; index < count; ++index) {
        if (static_cast<int32_t>(history_.size()) >= context_size_) break;
        const llama_token token = llama_sampler_sample(sampler_.get(), context_.get(), -1);
        if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab_, token)) break;
        // Proposing a token means decoding it here, so the next proposal
        // follows from it. The target may reject it, and the next sync()
        // rewinds this cache to whatever survived.
        llama_token mutable_token = token;
        if (llama_decode(context_.get(), llama_batch_get_one(&mutable_token, 1)) != 0) break;
        history_.push_back(token);
        proposals.push_back(token);
    }
    return proposals;
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
    if (input->config.n_batch > 0) {
        context_params.n_batch = static_cast<uint32_t>(input->config.n_batch);
    }
    if (input->config.n_ubatch > 0) {
        context_params.n_ubatch = static_cast<uint32_t>(input->config.n_ubatch);
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

    // How many handles may share one context and decode together. The draft
    // model is the one thing that rules batching out: verification reads
    // several positions of a batch this handle owns outright, which a shared
    // engine cannot promise. Such a handle gets a context to itself -- the
    // arrangement every handle had before batching existed.
    int32_t capacity = input->config.n_seq_max > 0 ? input->config.n_seq_max : 1;
    const bool wants_draft = input->draft_model_path && input->draft_model_path[0];
    if (wants_draft && capacity > 1) {
        UNIRT_LOG_INFO(
            "llama_cpp: n_seq_max={} ignored for a handle with a draft model; speculative "
            "decoding and batching cannot share a context",
            capacity);
        capacity = 1;
    }
    // Each sequence must end up with the window the caller asked for, and
    // llama.cpp splits n_ctx between them. Resolve "0 = whatever the model was
    // trained for" here so the multiplication has something to work with.
    int32_t per_sequence = input->config.n_ctx > 0
                               ? input->config.n_ctx
                               : static_cast<int32_t>(llama_model_n_ctx_train(model.get()));
    if (per_sequence <= 0) per_sequence = 4096;

    auto make_context = [&](int32_t seats) -> ContextPtr {
        llama_context_params params = context_params;
        params.n_seq_max            = static_cast<uint32_t>(seats);
        // One shared pool of KV cells rather than a fixed partition per
        // sequence. llama.cpp defaults the other way and its header suggests
        // splitting when the sequences do not share a prefix, but measured
        // here a split cache gives batching back everything it wins: four
        // sequences batched cost the same as four decoded separately (0.99x),
        // and unified they cost 1.25x less. The pool is sized for every
        // sequence at its full window, so sharing the cells cannot make one
        // conversation evict another's -- each handle still caps itself at
        // context_size().
        // One shared pool of KV cells rather than a fixed partition per
        // sequence, but only once there is something to share. llama.cpp
        // defaults the other way and its header suggests splitting when the
        // sequences do not share a prefix; measured here a split cache gives
        // most of batching back. Four sequences on CPU: 3.95x batched with a
        // split cache, 7.80x with a unified one. On Metal a split cache wins
        // nothing at all (0.99x against 1.27x).
        //
        // The cost is that a sequence's arithmetic now depends on what else
        // is in the pool -- the same request run alone and run in a batch can
        // part company on a near-tie, a token or two apart in the middle of a
        // sentence. That is the standard bargain for batched serving, it is
        // opt-in through n_seq_max, and a caller who needs the same tokens
        // every time asks for one sequence and gets a pool to itself.
        params.kv_unified = seats > 1;
        params.n_ctx                = static_cast<uint32_t>(per_sequence) *
                       static_cast<uint32_t>(seats);
        return ContextPtr(llama_init_from_model(model.get(), params));
    };

    // Handles group by everything that has to be identical for one context to
    // serve them all. n_threads is in there because it is a property of the
    // context, and two callers asking for different thread counts should not
    // silently get one of them.
    auto engine_key = [&](const std::string& device) {
        return device + '\x1f' + std::to_string(model_params.n_gpu_layers) + '\x1f' +
               std::to_string(per_sequence) + '\x1f' + std::to_string(capacity) + '\x1f' +
               std::to_string(context_params.n_batch) + '\x1f' +
               std::to_string(context_params.n_ubatch) + '\x1f' +
               std::to_string(context_params.n_threads) + '\x1f' +
               std::to_string(context_params.n_threads_batch) + '\x1f' + input->model_path;
    };

    int32_t      sequence = BatchEngine::kNoSequence;
    SharedEngine engine = acquire_engine(
        engine_key(device_key), capacity, per_sequence, make_context, model, sequence);
    if (!engine && (!input->device_id || !input->device_id[0]) &&
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
        if (model) {
            engine = acquire_engine(
                engine_key(selected_device_name), capacity, per_sequence, make_context, model,
                sequence);
        }
    }
    if (!engine) {
        UNIRT_LOG_ERROR("llama_cpp: failed to allocate inference context");
        return UNIRT_ERROR_COMMON_MODEL_LOAD;
    }

    if (engine->context_size() <= 0 || engine->batch_size() <= 0) {
        UNIRT_LOG_ERROR("llama_cpp: backend returned an invalid context configuration");
        engine->release(sequence);
        return UNIRT_ERROR_COMMON_MODEL_LOAD;
    }

    clear_model();
    engine_ = std::move(engine);
    sequence_ = sequence;
    // history_ mirrors this sequence's KV, which makes it the answer to
    // "what does slot N already hold" for a sibling looking to borrow a
    // prefix rather than evaluate it again.
    engine_->register_transcript(sequence_, &history_);
    vocab_ = llama_model_get_vocab(engine_->model().get());
    devices_ = std::move(selected_devices);
    template_override_ = std::move(requested_template);
    device_name_ = std::move(selected_device_name);
    context_size_ = engine_->context_size();
    {
        auto access = engine_->access();
        shift_supported_ = llama_memory_can_shift(access.memory()) &&
                           !llama_model_is_recurrent(engine_->model().get());
    }

    char description[256] = {};
    llama_model_desc(engine_->model().get(), description, sizeof(description));
    UNIRT_LOG_INFO(
        "llama_cpp: loaded {} [{}], context={}, batch={}, sequence={}/{}, device={}",
        input->model_path, description[0] ? description : "GGUF", context_size_,
        engine_->batch_size(), sequence_, engine_->capacity(), device_name_);

    if (input->draft_model_path && input->draft_model_path[0]) {
        // The draft gets the target's context size: it has to hold the same
        // prefix, and a shorter window would make it fail exactly when the
        // conversation got long enough for speculation to be worth anything.
        llama_context_params draft_context = context_params;
        draft_context.n_ctx = static_cast<uint32_t>(context_size_);
        llama_model_params draft_model_params = model_params;
        const int32_t loaded = draft_.load(
            input->draft_model_path, draft_model_params, draft_context, device_key, vocab_);
        if (loaded != UNIRT_SUCCESS) {
            // Refuse rather than quietly serve without it: a draft model that
            // was asked for and silently dropped looks like speculation that
            // does not help, which is a much harder thing to notice.
            clear_model();
            return loaded;
        }
    }
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    auto access = engine_->access();
    // This sequence only. The context may be holding three other conversations
    // that have nothing to do with this handle being reset.
    llama_memory_seq_rm(access.memory(), sequence_, -1, -1);
    access.truncated(sequence_, 0);
    history_.clear();
    last_logits_ = nullptr;
    llama_perf_context_reset(access.context());
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::save_kv_cache(
    const unirt_KvCacheSaveInput* input, unirt_KvCacheSaveOutput*) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !input->path || !input->path[0]) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    auto access = engine_->access();
    // Per sequence, not the whole context: what the caller means by "this
    // session" is this handle's conversation, and the context behind it may be
    // shared with three others.
    if (!llama_state_seq_save_file(
            access.context(), input->path, sequence_, history_.data(), history_.size())) {
        UNIRT_LOG_ERROR("llama_cpp: failed to save session {}", input->path);
        return UNIRT_ERROR_COMMON_UNKNOWN;
    }
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::load_kv_cache(
    const unirt_KvCacheLoadInput* input, unirt_KvCacheLoadOutput*) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !input->path || !input->path[0]) return UNIRT_ERROR_COMMON_INVALID_INPUT;

    std::vector<llama_token> loaded(static_cast<size_t>(context_size_));
    size_t                   count = 0;
    auto                     access = engine_->access();
    access.truncated(sequence_, 0);
    last_logits_ = nullptr;
    if (!llama_state_seq_load_file(
            access.context(), input->path, sequence_, loaded.data(), loaded.size(), &count)) {
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
    if (!engine_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
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
                           ? llama_model_chat_template(engine_->model().get(), nullptr)
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
    // Both conditions, and the flag is the caller's. Evicting without being
    // asked drops the oldest turns and answers from a conversation the caller
    // never sent -- silently, since the reply looks perfectly normal. The ABI
    // says overflow is an error unless sliding_window says otherwise, and
    // this backend used to ignore that and shift regardless.
    if (!shift_supported_ || !sliding_window_) return 0;
    const int32_t cached = static_cast<int32_t>(history_.size());
    // Past the pinned head, and past anything a sibling sequence borrowed:
    // llama.cpp renumbers a cell for every sequence holding it, so shifting
    // inside a shared region would move somebody else's tokens.
    const int32_t head =
        std::min(std::max(pinned_head_, engine_->shared_prefix(sequence_)), cached);
    // Free at least enough for the incoming tokens, but take half of the
    // evictable region so we do not shift on every subsequent token.
    const int32_t deficit  = cached + incoming - context_size_ + 1;
    int32_t       n_evict  = std::max((cached - head) / 2, deficit);
    n_evict                = std::min(n_evict, cached - head);
    if (n_evict <= 0) return 0;

    {
        auto access = engine_->access();
        llama_memory_seq_rm(access.memory(), sequence_, head, head + n_evict);
        llama_memory_seq_add(access.memory(), sequence_, head + n_evict, cached, -n_evict);
        access.truncated(sequence_, cached - n_evict);
    }
    history_.erase(history_.begin() + head, history_.begin() + head + n_evict);
    UNIRT_LOG_INFO(
        "llama_cpp: context shift evicted {} tokens (head {}, cached {} -> {})", n_evict, head,
        cached, history_.size());
    return n_evict;
}

int32_t LlamaCppLlm::decode(const llama_token* tokens, int32_t count) {
    if (count <= 0) return UNIRT_SUCCESS;
    last_logits_ = nullptr;
    // A submission is one round, so a long prompt submitted whole is a round
    // the other sequences cannot get into until it finishes -- three streams
    // stopped dead for 2.0 s on CPU while a fourth request's 2000-token prompt
    // went through. Chunking it to the physical batch size interleaves them:
    // each round is one ubatch of the prompt plus everyone else's next token.
    //
    // n_ubatch and not a number of our own. Below it the prompt is split into
    // rounds finer than the backend's own unit of work, which costs round
    // overhead and buys nothing; above it, ubatches llama.cpp was going to run
    // separately anyway are welded into a single round that excludes everyone
    // else. Measured over 64..2044 with a 2000-token prompt arriving beside
    // three streams, the knee sits exactly there. At n_ubatch (512) the worst
    // gap a decoder sees falls from 1650 ms to 600 ms on CPU and from 440 ms
    // to 145 ms on Metal, and the big request's own latency rises 4-5%.
    // Halving the chunk again takes CPU to 370 ms but costs 20% of that
    // latency.
    //
    // The decoders lose slightly more time in total this way -- four rounds
    // carrying a chunk cost more than one round carrying the lot. What they
    // get is that none of it arrives at once: a stream hitches four times for
    // half a second instead of stopping dead for nearly two.
    const int32_t batch_size = engine_->batch_size();
    int32_t chunk_limit =
        engine_->exclusive()
            ? batch_size
            : std::max(1, std::min(batch_size - engine_->capacity(), engine_->ubatch_size()));
    if (const char* tune = std::getenv("UNIRT_PREFILL_CHUNK")) {
        chunk_limit = std::max(1, std::min(batch_size, std::atoi(tune)));
    }
    for (int32_t offset = 0; offset < count; offset += chunk_limit) {
        const int32_t chunk_size = std::min(chunk_limit, count - offset);
        const llama_pos position = static_cast<llama_pos>(history_.size());
        DecodeStatus status = engine_->decode(
            sequence_, tokens + offset, chunk_size, position, &last_logits_);
        // "No KV slot" means this sequence is full; shift its window and retry
        // instead of failing, as long as eviction keeps making progress. A
        // rolled-back round is not a failure at all -- resubmit unchanged.
        while (status == DecodeStatus::retry ||
               (status == DecodeStatus::no_kv_slot && evict_for_space(chunk_size) > 0)) {
            status = engine_->decode(
                sequence_, tokens + offset, chunk_size,
                static_cast<llama_pos>(history_.size()), &last_logits_);
        }
        if (status == DecodeStatus::ok) {
            // history_ mirrors the KV contents at all times so eviction and
            // prefix reuse can reason about positions from it alone.
            history_.insert(history_.end(), tokens + offset, tokens + offset + chunk_size);
            continue;
        }
        UNIRT_LOG_ERROR(
            "llama_cpp: decode failed with code {}", static_cast<int32_t>(status));
        if (status == DecodeStatus::no_kv_slot) {
            return UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
        }
        if (status == DecodeStatus::aborted) return UNIRT_ERROR_COMMON_CANCELLED;
        return UNIRT_ERROR_LLM_GENERATION_FAILED;
    }
    return UNIRT_SUCCESS;
}

llama_token LlamaCppLlm::pick(llama_sampler* sampler, const float* logits) const {
    if (!logits) return LLAMA_TOKEN_NULL;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab_);

    candidates_.resize(static_cast<size_t>(n_vocab));
    for (int32_t token = 0; token < n_vocab; ++token) {
        candidates_[static_cast<size_t>(token)] = llama_token_data{token, logits[token], 0.0f};
    }
    llama_token_data_array array = {candidates_.data(), candidates_.size(), -1, false};
    llama_sampler_apply(sampler, &array);
    if (array.selected < 0 || array.selected >= static_cast<int64_t>(array.size)) {
        return LLAMA_TOKEN_NULL;
    }
    return array.data[array.selected].id;
}

int32_t LlamaCppLlm::speculate(
    llama_sampler* sampler, int32_t budget, std::vector<llama_token>& accepted, bool& spoiled) {
    spoiled = false;
    accepted.clear();
    // Only ever reached on an engine of capacity one (create() forces that
    // whenever a draft model is attached), so the context and its output
    // buffer belong to this handle alone and reading several positions of the
    // verification batch back out of it needs no round.
    llama_context* context = engine_->unshared_context();
    if (!context) return UNIRT_ERROR_LLM_GENERATION_FAILED;

    const int32_t proposal_count = std::min(budget - 1, draft_tokens_);
    std::vector<llama_token> proposals;
    if (proposal_count > 0 && draft_.sync(history_) == UNIRT_SUCCESS) {
        proposals = draft_.propose(proposal_count);
    }

    // The token the target would have produced anyway, from the logits it
    // already has. Sampled without accepting: if it disagrees with the first
    // proposal, everything after it is discarded and only this one is kept.
    const llama_token first = pick(sampler, last_logits_);
    if (first == LLAMA_TOKEN_NULL) return UNIRT_ERROR_LLM_GENERATION_FAILED;
    accepted.push_back(first);
    // Accept as we go, not at the end: the next position is picked with the
    // penalty and grammar state that this token puts the chain in, which is
    // what the non-speculative path would have done at the same point.
    llama_sampler_accept(sampler, first);
    if (proposals.empty() || proposals[0] != first) {
        // Nothing to verify, or the very first guess was wrong. Either way the
        // caller decodes `first` itself, exactly as it would without a draft.
        draft_proposed_ += static_cast<int64_t>(proposals.size());
        return UNIRT_SUCCESS;
    }

    // The target sees the agreed token plus every proposal after it, in one
    // batch with logits at each position: position i answers "what follows
    // proposals[0..i]". That single pass is the whole saving.
    const int32_t batch_count = static_cast<int32_t>(proposals.size());
    if (history_.size() + static_cast<size_t>(batch_count) >
        static_cast<size_t>(context_size_)) {
        draft_proposed_ += static_cast<int64_t>(proposals.size());
        return UNIRT_SUCCESS;
    }
    llama_batch batch = llama_batch_init(batch_count, 0, 1);
    for (int32_t index = 0; index < batch_count; ++index) {
        batch.token[index]    = proposals[static_cast<size_t>(index)];
        batch.pos[index]      = static_cast<llama_pos>(history_.size()) + index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = sequence_;
        batch.logits[index]   = 1;
    }
    batch.n_tokens = batch_count;
    const int32_t decoded = llama_decode(context, batch);
    llama_batch_free(batch);
    if (decoded != 0) {
        // The KV may hold part of the batch; llama.cpp restores it on a
        // negative return, but rc==1 (no slot) leaves nothing decoded. Either
        // way the safe move is to say so and let the caller decode one token.
        spoiled = decoded > 0;
        if (decoded < 0) return UNIRT_ERROR_LLM_GENERATION_FAILED;
        return UNIRT_SUCCESS;
    }
    history_.insert(history_.end(), proposals.begin(), proposals.end());

    // proposals[0] is already agreed. Walk the rest: the target's answer at
    // position i-1 is what it would have said after proposals[0..i-1].
    int32_t matched = 1;
    llama_token target_next = LLAMA_TOKEN_NULL;
    while (matched < batch_count) {
        target_next = pick(sampler, llama_get_logits_ith(context, matched - 1));
        if (target_next == LLAMA_TOKEN_NULL) return UNIRT_ERROR_LLM_GENERATION_FAILED;
        if (target_next != proposals[static_cast<size_t>(matched)]) break;
        accepted.push_back(target_next);
        llama_sampler_accept(sampler, target_next);
        ++matched;
        target_next = LLAMA_TOKEN_NULL;
    }
    if (matched == batch_count) {
        // Every proposal survived, so the last position's logits are a free
        // extra token: the target evaluated it in the same pass.
        target_next = pick(sampler, llama_get_logits_ith(context, batch_count - 1));
        if (target_next == LLAMA_TOKEN_NULL) return UNIRT_ERROR_LLM_GENERATION_FAILED;
    }

    // Whatever the target rejected is still in its KV. Roll back to the agreed
    // prefix; the caller decodes target_next on top of it, which puts the
    // divergent token in the right position.
    const size_t keep = history_.size() - static_cast<size_t>(batch_count - matched);
    if (keep < history_.size()) {
        auto* memory = llama_get_memory(context);
        if (!llama_memory_seq_rm(memory, sequence_, static_cast<llama_pos>(keep), -1)) {
            llama_memory_seq_rm(memory, sequence_, -1, -1);
            history_.clear();
            spoiled = true;
            return UNIRT_SUCCESS;
        }
        history_.resize(keep);
    }
    accepted.push_back(target_next);
    llama_sampler_accept(sampler, target_next);

    draft_proposed_ += static_cast<int64_t>(proposals.size());
    draft_accepted_ += matched;
    // The accepted proposals are in the KV already (they were part of the
    // batch and survived the rollback); the caller must not decode those
    // again, only the last entry. Report that by leaving history_ at `keep`,
    // which is exactly accepted.size() - 1 tokens ahead of where it started.
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
    if (!engine_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
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
        // 0 keeps the plugin default; negative turns speculation off for this
        // request without unloading the draft model.
        if (input->config->n_draft != 0) draft_tokens_ = input->config->n_draft;
        sliding_window_ = input->config->sliding_window;
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
        auto access = engine_->access();
        access.truncated(sequence_, requested_past);
        if (!llama_memory_seq_rm(access.memory(), sequence_, requested_past, -1)) {
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
    // Before deciding what of our own cache survives, see whether a sibling
    // sequence is holding more of this prompt than we are -- four clients
    // arriving with the same system prompt is the ordinary shape of serving,
    // and evaluating it once per client is the ordinary waste. Skipped when
    // the caller is driving positions itself: n_past > 0 says it knows what
    // is in the cache, and replacing that underneath it would be a surprise.
    if (requested_past == 0) {
        const size_t borrowed = engine_->borrow_prefix(
            sequence_, prompt, reusable_prefix(prompt));
        if (borrowed > 0) {
            history_.assign(prompt.begin(), prompt.begin() + static_cast<ptrdiff_t>(borrowed));
            last_logits_ = nullptr;
        }
    }

    const size_t shared = reusable_prefix(prompt);
    if (shared < history_.size()) {
        auto access = engine_->access();
        access.truncated(sequence_, static_cast<int32_t>(shared));
        last_logits_ = nullptr;
        if (shared == 0) {
            llama_memory_seq_rm(access.memory(), sequence_, -1, -1);
        } else if (!llama_memory_seq_rm(
                       access.memory(), sequence_, static_cast<llama_pos>(shared), -1)) {
            llama_memory_seq_rm(access.memory(), sequence_, -1, -1);
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
    if ((!shift_supported_ || !sliding_window_) &&
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

    // From here to the end of the call this sequence is one the engine should
    // hold a round open for. Nothing above this point submits anything.
    struct Membership {
        BatchEngine& engine;
        int32_t      sequence;
        Membership(BatchEngine& e, int32_t s) : engine(e), sequence(s) { engine.join(sequence); }
        ~Membership() { engine.leave(sequence); }
    } membership(*engine_, sequence_);

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

    // Speculation is off for grammars and for logprobs, and neither is a
    // limitation of the algorithm. A grammar rejects most of what an
    // unconstrained draft proposes, so the verification batch is wasted work.
    // Logprobs are read from the logits of the step that produced the token,
    // and a verified batch overwrites the first one's before anybody looks --
    // reporting the wrong distribution would be worse than not speculating.
    const bool speculating =
        draft_.ready() && draft_tokens_ > 0 && !sampler_has_grammar && !reporter;

    std::vector<llama_token> step;
    bool done = false;
    while (!done && generated < max_tokens) {
        if (history_.size() >= static_cast<size_t>(context_size_) && evict_for_space(1) == 0) {
            stop_reason = "context_length";
            if (!stream_state.emit_safe(input->on_token, input->user_data, true)) {
                stream_state.discard_unemitted();
                stop_reason = "user";
            }
            break;
        }

        // Every token in `step` but the last is already in the KV: the target
        // decoded them as part of the verification batch and kept them.
        size_t already_decoded = 0;
        step.clear();
        if (speculating) {
            bool spoiled = false;
            const size_t before = history_.size();
            result = speculate(sampler.get(), max_tokens - generated, step, spoiled);
            if (result != UNIRT_SUCCESS) break;
            if (spoiled) step.clear();
            already_decoded = history_.size() - before;
        }
        if (step.empty()) {
            // pick() rather than llama_sampler_sample(): the scores come from
            // the engine's own copy of this sequence's row, which a batch-mate's
            // round cannot have overwritten, and accepting is done here so the
            // chain advances exactly once per kept token.
            const llama_token token = pick(sampler.get(), last_logits_);
            if (token == LLAMA_TOKEN_NULL) {
                result = UNIRT_ERROR_LLM_GENERATION_FAILED;
                break;
            }
            llama_sampler_accept(sampler.get(), token);
            step.push_back(token);
            already_decoded = 0;
        }

        for (size_t index = 0; index < step.size() && !done; ++index) {
            const llama_token token = step[index];
            if (llama_vocab_is_eog(vocab_, token)) {
                stop_reason = "eos";
                if (!stream_state.emit_safe(input->on_token, input->user_data, true)) {
                    stream_state.discard_unemitted();
                    stop_reason = "user";
                }
                done = true;
                break;
            }

            std::string piece = token_piece(token);
            if (generated == 0) first_token_time = monotonic_us();

            // Before decode(): the logits still belong to the step that
            // produced this token. Decoding it overwrites them with the next
            // step's. (Only ever reached without speculation -- see above.)
            if (reporter && !reporter->report(
                                last_logits_, token, input->on_logprob, input->user_data)) {
                stream_state.discard_unemitted();
                stop_reason = "user";
                done = true;
                break;
            }

            // Commit every non-EOG sampled token before exposing it.  This
            // keeps llama.cpp's KV state and history_ identical even for
            // max-length, callback cancellation, and stop-sequence exits.
            // decode() itself appends the token to history_.
            if (index >= already_decoded) {
                result = decode(&token, 1);
                if (result != UNIRT_SUCCESS) {
                    done = true;
                    break;
                }
            }

            stream_state.append(piece);
            ++generated;
            const bool hit_stop = stream_state.find_and_trim_stop();
            if (!stream_state.emit_safe(input->on_token, input->user_data, hit_stop)) {
                stream_state.discard_unemitted();
                stop_reason = "user";
                done = true;
                break;
            }
            if (hit_stop) {
                stop_reason = "stop_sequence";
                done = true;
                break;
            }
            if (generated >= max_tokens) {
                if (!stream_state.emit_safe(input->on_token, input->user_data, true)) {
                    stream_state.discard_unemitted();
                    stop_reason = "user";
                }
                done = true;
                break;
            }
        }
    }
    if (speculating && draft_proposed_ > 0) {
        UNIRT_LOG_DEBUG(
            "llama_cpp: draft acceptance {}/{} tokens ({}%)", draft_accepted_, draft_proposed_,
            100 * draft_accepted_ / draft_proposed_);
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
    if (!engine_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    output->vocab_size = llama_vocab_n_tokens(vocab_);
    const llama_token bos = llama_vocab_bos(vocab_);
    output->bos_token = bos == LLAMA_TOKEN_NULL ? -1 : bos;
    output->add_bos = llama_vocab_get_add_bos(vocab_) ? 1 : 0;
    return UNIRT_SUCCESS;
}

int32_t LlamaCppLlm::get_runtime_stats(unirt_LlmRuntimeStats* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    output->model_bytes = static_cast<int64_t>(llama_model_size(engine_->model().get()));
    // This sequence's share, not the pool's: with four handles on one context
    // the pool figure would be the same number four times over and would look
    // like four times the memory.
    auto access = engine_->access();
    output->kv_cache_bytes =
        static_cast<int64_t>(llama_state_seq_get_size(access.context(), sequence_));
    output->device_peak_bytes = -1;
    output->device_name = device_name_.c_str();
    return UNIRT_SUCCESS;
}

}  // namespace unirt::llama_plugin

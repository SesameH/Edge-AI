// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "plugin/plugin_export.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

#include "build_config.h"
#include "generation_state.h"
#include "json_constraint.h"
#include "logging.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

// MLX backend: runs Llama-architecture safetensors models (HF layout:
// config.json + tokenizer.json + model.safetensors in one directory) on
// the Apple Silicon GPU through MLX. Greedy decoding in v1.

namespace mx = mlx::core;

namespace unirt::mlx_plugin {

namespace {

char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out) std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool valid_sampler_config(const unirt_SamplerConfig& config) {
    return std::isfinite(config.temperature) && config.temperature >= 0.f && config.temperature <= 2.f &&
           std::isfinite(config.top_p) && config.top_p >= 0.f && config.top_p <= 1.f &&
           config.top_k >= 0 &&
           std::isfinite(config.min_p) && config.min_p >= 0.f && config.min_p <= 1.f &&
           std::isfinite(config.repetition_penalty) && config.repetition_penalty >= 0.f &&
           std::isfinite(config.presence_penalty) &&
           std::isfinite(config.frequency_penalty);
}

bool metal_available() noexcept {
    try {
        if (!mx::is_available(mx::Device(mx::Device::gpu))) return false;
        // A Metal backend can be compiled in while the current process has no
        // actual device (headless sandbox/VM). device_info forces that final
        // availability check and throws in exactly that situation.
        (void) mx::device_info(mx::Device(mx::Device::gpu));
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

class MlxLlm : public LlmBackend {
   public:
    int32_t create(const unirt_LlmCreateInput* input) override {
        if (!input || !input->model_path) {
            UNIRT_LOG_ERROR("mlx: create input/model_path is null");
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        if (input->device_id && input->device_id[0] &&
            std::strcmp(input->device_id, "mlx0") != 0) {
            UNIRT_LOG_ERROR("mlx: unknown device id '{}'", input->device_id);
            return UNIRT_ERROR_COMMON_INVALID_DEVICE;
        }
        namespace fs = std::filesystem;
        fs::path model_dir(input->model_path);
        if (fs::is_regular_file(model_dir)) model_dir = model_dir.parent_path();
        bool has_tensors = fs::is_regular_file(model_dir / "model.safetensors") ||
                           fs::is_regular_file(model_dir / "model.safetensors.index.json");
        if (!has_tensors && fs::is_directory(model_dir)) {
            for (const auto& entry : fs::directory_iterator(model_dir)) {
                const std::string name = entry.path().filename().string();
                if (entry.is_regular_file() && name.rfind("model-", 0) == 0 &&
                    entry.path().extension() == ".safetensors") {
                    has_tensors = true;
                    break;
                }
            }
        }
        if (!fs::is_regular_file(model_dir / "config.json") || !has_tensors) {
            UNIRT_LOG_ERROR("mlx: {} must contain config.json and model safetensors", model_dir.string());
            return UNIRT_ERROR_COMMON_FILE_NOT_FOUND;
        }
        fs::path tok_path = (input->tokenizer_path && input->tokenizer_path[0])
                                ? fs::path(input->tokenizer_path)
                                : model_dir / "tokenizer.json";
        if (!fs::is_regular_file(tok_path)) {
            UNIRT_LOG_ERROR("mlx: tokenizer file does not exist: {}", tok_path.string());
            return UNIRT_ERROR_COMMON_FILE_NOT_FOUND;
        }
        try {
            auto cfg = LlamaConfig::from_json_file((model_dir / "config.json").string());
            tokenizer_.load(tok_path.string());
            if (tokenizer_.vocab_size() != cfg.vocab_size) {
                throw std::runtime_error(
                    "tokenizer/model vocabulary size mismatch (" +
                    std::to_string(tokenizer_.vocab_size()) + " vs " +
                    std::to_string(cfg.vocab_size) + ")");
            }
            if (!metal_available()) {
                UNIRT_LOG_ERROR("mlx: no usable Metal device is available");
                return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
            }
            mx::set_default_device(mx::Device(mx::Device::gpu));
            model_.load(model_dir.string(), cfg);
        } catch (const std::exception& e) {
            UNIRT_LOG_ERROR("mlx: model load failed: {}", e.what());
            return UNIRT_ERROR_COMMON_MODEL_LOAD;
        }
        n_ctx_ = input->config.n_ctx > 0 ? input->config.n_ctx
                                         : model_.config().max_position_embeddings;
        if (n_ctx_ <= 0 || n_ctx_ > model_.config().max_position_embeddings) {
            UNIRT_LOG_ERROR(
                "mlx: requested context {} exceeds model maximum {}", n_ctx_,
                model_.config().max_position_embeddings);
            model_.reset_cache();
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        model_.reserve_cache(n_ctx_);
        transcript_.clear();
        UNIRT_LOG_INFO("mlx: loaded {} ({} layers, vocab {}, ctx {})", model_dir.string(),
                        model_.config().num_hidden_layers, model_.config().vocab_size, n_ctx_);
        return UNIRT_SUCCESS;
    }

    int32_t reset() override {
        model_.reset_cache();
        transcript_.clear();
        return UNIRT_SUCCESS;
    }

    int32_t save_kv_cache(const unirt_KvCacheSaveInput*, unirt_KvCacheSaveOutput*) override {
        return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    int32_t load_kv_cache(const unirt_KvCacheLoadInput*, unirt_KvCacheLoadOutput*) override {
        return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }

    int32_t apply_chat_template(
        const unirt_LlmApplyChatTemplateInput* input, unirt_LlmApplyChatTemplateOutput* output) override {
        if (!input || !output || (!input->messages && input->message_count > 0)) {
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        // ChatML (SmolLM2 / Qwen convention).
        std::string out;
        for (int32_t i = 0; i < input->message_count; ++i) {
            const auto& m = input->messages[i];
            out += "<|im_start|>";
            out += m.role ? m.role : "user";
            out += "\n";
            out += m.content ? m.content : "";
            out += "<|im_end|>\n";
        }
        if (input->add_generation_prompt) out += "<|im_start|>assistant\n";
        output->formatted_text = dup_cstr(out);
        return output->formatted_text ? UNIRT_SUCCESS : UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
    }

    int32_t generate(const unirt_LlmGenerateInput* input, unirt_LlmGenerateOutput* output) override {
        if (!input || !output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
        *output = {};

        int32_t max_tokens = 512;
        int32_t requested_past = 0;
        std::vector<std::string> stops;
        const unirt_SamplerConfig* sampler_config = nullptr;
        if (input->config) {
            if (input->config->max_tokens < 0 || input->config->stop_count < 0 ||
                (input->config->stop_count > 0 && !input->config->stop)) {
                return UNIRT_ERROR_COMMON_INVALID_INPUT;
            }
            if (input->config->max_tokens > 0) max_tokens = input->config->max_tokens;
            requested_past = input->config->n_past;
            sampler_config = input->config->sampler_config;
            if (input->config->sliding_window) {
                return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
            }
            for (int32_t i = 0; i < input->config->stop_count; ++i) {
                if (input->config->stop[i] && input->config->stop[i][0]) {
                    stops.emplace_back(input->config->stop[i]);
                }
            }
        }
        if (requested_past < 0 || requested_past > model_.n_past()) {
            UNIRT_LOG_ERROR(
                "mlx: n_past={} is invalid for {} cached tokens", requested_past,
                model_.n_past());
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        // n_past > 0 is an explicit rewind; n_past == 0 defers to the
        // prefix-reuse pass below.
        if (requested_past > 0 && requested_past < model_.n_past()) {
            if (!model_.trim_cache(requested_past)) {
                return UNIRT_ERROR_COMMON_INVALID_INPUT;
            }
            transcript_.resize(static_cast<size_t>(requested_past));
        }
        if (sampler_config && !valid_sampler_config(*sampler_config)) {
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        // GBNF is llama.cpp's grammar dialect and has no compiler here; JSON
        // schema and JSON mode do (json_constraint.h).
        if (sampler_config &&
            ((sampler_config->grammar_path && sampler_config->grammar_path[0]) ||
             (sampler_config->grammar_string && sampler_config->grammar_string[0]))) {
            UNIRT_LOG_ERROR("mlx: GBNF grammars are not supported; use json_schema instead");
            return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
        }
        std::unique_ptr<JsonConstraint> constraint;
        if (sampler_config &&
            (sampler_config->enable_json ||
             (sampler_config->json_schema && sampler_config->json_schema[0]))) {
            const std::string schema =
                sampler_config->json_schema ? sampler_config->json_schema : std::string();
            std::string error;
            constraint = JsonConstraint::compile(schema, error);
            if (!constraint) {
                UNIRT_LOG_ERROR("mlx: {}", error);
                return UNIRT_ERROR_COMMON_INVALID_INPUT;
            }
            mask_cache_.clear();  // masks belong to one schema, not to the model
        }

        std::vector<int32_t> prompt_ids;
        if (input->input_ids && input->input_ids_count > 0) {
            prompt_ids.assign(input->input_ids, input->input_ids + input->input_ids_count);
        } else if (input->prompt_utf8) {
            try {
                prompt_ids = tokenizer_.encode(input->prompt_utf8);
            } catch (const std::exception& error) {
                UNIRT_LOG_ERROR("mlx: tokenization failed: {}", error.what());
                return UNIRT_ERROR_LLM_TOKENIZATION_FAILED;
            }
        } else {
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        if (prompt_ids.empty()) return UNIRT_ERROR_COMMON_INVALID_INPUT;
        for (int32_t token : prompt_ids) {
            if (token < 0 || token >= model_.config().vocab_size) {
                UNIRT_LOG_ERROR("mlx: input token id {} is outside the vocabulary", token);
                return UNIRT_ERROR_COMMON_INVALID_INPUT;
            }
        }
        // Reuse the KV prefix shared with the previous transcript instead of
        // re-evaluating it; keep at least one prompt token for fresh logits.
        size_t       shared = 0;
        const size_t reuse_cap = std::min(transcript_.size(), prompt_ids.size() - 1);
        while (shared < reuse_cap && transcript_[shared] == prompt_ids[shared]) ++shared;
        if (shared < transcript_.size()) {
            if (!model_.trim_cache(static_cast<int32_t>(shared))) {
                model_.reset_cache();
                transcript_.clear();
                shared = 0;
            } else {
                transcript_.resize(shared);
            }
        }
        std::vector<int32_t> fresh_ids(prompt_ids.begin() + shared, prompt_ids.end());
        if (shared > 0) {
            UNIRT_LOG_DEBUG("mlx: prefix cache reused {} tokens, evaluating {}", shared, fresh_ids.size());
        }

        if (fresh_ids.size() > static_cast<size_t>(n_ctx_ - model_.n_past())) {
            if (!shift_context() ||
                fresh_ids.size() > static_cast<size_t>(n_ctx_ - model_.n_past())) {
                UNIRT_LOG_ERROR("mlx: context length {} exceeded", n_ctx_);
                return UNIRT_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
            }
        }

        const int32_t eos = tokenizer_.eos_id() >= 0 ? tokenizer_.eos_id() : model_.config().eos_token_id;

        Sampler sampler(sampler_config);
        for (int32_t id : prompt_ids) sampler.observe(id);

        StopStreamState stream_state(std::move(stops));
        const char* stop_reason = "length";

        const int64_t t_start = now_us();
        auto          logits  = model_.forward_logits(fresh_ids);
        transcript_.insert(transcript_.end(), fresh_ids.begin(), fresh_ids.end());
        bool exhausted = constraint && !constrain_logits(logits, *constraint, eos);
        int32_t       next    = exhausted ? eos : sampler.sample(logits);
        const int64_t t_first = now_us();

        int32_t generated = 0;
        while (generated < max_tokens) {
            if (next == eos || exhausted) {
                stop_reason = exhausted ? "length" : "eos";
                if (!stream_state.emit_safe(input->on_token, input->user_data, true)) {
                    stream_state.discard_unemitted();
                    stop_reason = "user";
                }
                break;
            }
            if (model_.n_past() >= n_ctx_ && !shift_context()) {
                stop_reason = "context_length";
                if (!stream_state.emit_safe(input->on_token, input->user_data, true)) {
                    stream_state.discard_unemitted();
                    stop_reason = "user";
                }
                break;
            }
            std::string piece = tokenizer_.decode_piece(next);
            if (constraint && !constraint->accept(piece)) {
                // Unreachable unless masking and the automaton disagree.
                UNIRT_LOG_ERROR("mlx: sampled token {} violates the JSON constraint", next);
                return UNIRT_ERROR_LLM_GENERATION_FAILED;
            }

            // Commit the sampled token to every layer's KV cache before any
            // exit path.  The returned logits are used only when decoding
            // continues, but evaluating them makes n_past/KV exact.
            sampler.observe(next);
            logits = model_.forward_logits({next});
            transcript_.push_back(next);
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
            exhausted = constraint && !constrain_logits(logits, *constraint, eos);
            next      = exhausted ? eos : sampler.sample(logits);
        }
        const int64_t t_end = now_us();

        output->full_text = dup_cstr(stream_state.text());

        auto& p            = output->profile_data;
        p                  = {};
        p.prompt_tokens    = static_cast<int64_t>(prompt_ids.size());
        p.generated_tokens = generated;
        p.ttft             = t_first - t_start;
        p.prompt_time      = t_first - t_start;
        p.decode_time      = t_end - t_first;
        p.prefill_speed    = p.prompt_time > 0 ? p.prompt_tokens * 1e6 / p.prompt_time : 0.0;
        p.decoding_speed   = p.decode_time > 0 ? p.generated_tokens * 1e6 / p.decode_time : 0.0;
        p.stop_reason      = stop_reason;

        return output->full_text ? UNIRT_SUCCESS : UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
    }

    int32_t get_model_info(unirt_LlmModelInfo* info) override {
        if (!info) return UNIRT_ERROR_COMMON_INVALID_INPUT;
        info->vocab_size = model_.config().vocab_size;
        info->bos_token  = model_.config().bos_token_id;
        info->add_bos    = 0;  // ChatML models embed specials in the template
        return UNIRT_SUCCESS;
    }

    int32_t get_runtime_stats(unirt_LlmRuntimeStats* output) override {
        if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
        output->model_bytes       = model_.weights_bytes();
        output->kv_cache_bytes    = model_.kv_cache_bytes();
        output->device_peak_bytes = static_cast<int64_t>(mx::get_peak_memory());
        static std::string dev_name;
        if (dev_name.empty()) {
            dev_name = "Apple Silicon GPU (Metal/MLX)";
        }
        output->device_name = dev_name.c_str();
        return UNIRT_SUCCESS;
    }

   private:
    // Evicts the older half of the cached transcript to make room when the
    // context fills up (mirrors llama_cpp's context shifting, per
    // CLAUDE.md). MLX has no in-place "shift cached RoPE positions" op
    // exposed, so instead of rotating the existing KV cache this discards
    // the oldest tokens and re-runs forward_logits() over the retained
    // tail — identical math to a fresh prefill of that tail, just paid
    // again at eviction time instead of never. Returns false only if there
    // is nothing left to keep (n_ctx_ <= 1).
    bool shift_context() {
        if (n_ctx_ <= 1 || transcript_.empty()) return false;
        const size_t keep = std::min(transcript_.size(), static_cast<size_t>(std::max(1, n_ctx_ / 2)));
        std::vector<int32_t> retained(transcript_.end() - static_cast<ptrdiff_t>(keep), transcript_.end());
        model_.reset_cache();
        model_.forward_logits(retained);
        transcript_.assign(retained.begin(), retained.end());
        UNIRT_LOG_DEBUG("mlx: context shifted, retained {} of {} token budget", keep, n_ctx_);
        return true;
    }

    // Constrained decoding tests every candidate token against the grammar on
    // every step, so the per-id byte decode is hoisted out of that loop and
    // done once per model. Tokens are also bucketed by first byte: the schema
    // usually admits only a handful of next bytes, and a bucket whose first
    // byte is already refused cannot contain an acceptable token.
    void build_vocabulary_cache() {
        if (!token_bytes_.empty()) return;
        const size_t count = static_cast<size_t>(tokenizer_.vocab_size());
        token_bytes_.resize(count);
        first_byte_buckets_.assign(256, {});
        for (size_t id = 0; id < count; ++id) {
            const int32_t token = static_cast<int32_t>(id);
            // Special tokens are control markers, not JSON text; EOS is
            // handled separately and the rest must never appear mid-value.
            if (tokenizer_.is_special(token)) continue;
            token_bytes_[id] = tokenizer_.decode_piece(token);
            if (token_bytes_[id].empty()) continue;
            const auto first = static_cast<unsigned char>(token_bytes_[id][0]);
            first_byte_buckets_[first].push_back(token);
        }
    }

    // Sets every token the schema forbids to -inf. Returns false when nothing
    // at all may follow, which can only happen if the schema admits no
    // continuation the tokenizer can spell.
    bool constrain_logits(std::vector<float>& logits, JsonConstraint& constraint, int32_t eos) {
        build_vocabulary_cache();
        // Testing 49k tokens against the automaton costs about as much as the
        // forward pass on a small model, and most steps repeat a state that
        // was already scored -- every character inside a string leaves the
        // automaton in the same place. Keyed on the state, the mask is
        // computed a handful of times per generation instead of per token.
        const std::string signature = constraint.state_signature();
        auto              cached    = mask_cache_.find(signature);
        if (cached == mask_cache_.end()) {
            std::vector<uint8_t> allowed(logits.size(), 0);
            bool                 any = false;
            for (int byte = 0; byte < 256; ++byte) {
                const auto& bucket = first_byte_buckets_[static_cast<size_t>(byte)];
                if (bucket.empty()) continue;
                if (!constraint.allows(std::string(1, static_cast<char>(byte)))) continue;
                for (int32_t token : bucket) {
                    if (static_cast<size_t>(token) >= allowed.size()) continue;
                    if (!constraint.allows(token_bytes_[static_cast<size_t>(token)])) continue;
                    allowed[static_cast<size_t>(token)] = 1;
                    any                                 = true;
                }
            }
            // Ending the value is the grammar's decision, not the model's:
            // EOS stays masked until the JSON is syntactically complete.
            if (eos >= 0 && static_cast<size_t>(eos) < allowed.size() && constraint.can_stop()) {
                allowed[static_cast<size_t>(eos)] = 1;
                any                               = true;
            }
            // An empty mask is the "nothing may follow" marker, cached like
            // any other so a dead end is not rediscovered token by token.
            if (!any) allowed.clear();
            if (mask_cache_.size() >= kMaxCachedMasks) mask_cache_.clear();
            cached = mask_cache_.emplace(signature, std::move(allowed)).first;
        }
        const std::vector<uint8_t>& allowed = cached->second;
        if (allowed.empty()) return false;
        for (size_t id = 0; id < logits.size(); ++id) {
            if (id >= allowed.size() || !allowed[id]) {
                logits[id] = -std::numeric_limits<float>::infinity();
            }
        }
        return true;
    }

    BpeTokenizer tokenizer_;
    LlamaModel   model_;
    int32_t      n_ctx_ = 0;
    std::vector<std::string>          token_bytes_;
    std::vector<std::vector<int32_t>> first_byte_buckets_;
    // Bounded so a long generation over a deep schema cannot grow it without
    // limit; a schema that busy is also one whose states rarely repeat.
    static constexpr size_t                                 kMaxCachedMasks = 64;
    std::unordered_map<std::string, std::vector<uint8_t>>   mask_cache_;
    // Ids of every token currently represented in the KV cache, in order.
    // Lets a resent conversation transcript skip re-evaluating its prefix.
    std::vector<int32_t> transcript_;
};

class MlxPlugin : public BackendPackage {
   public:
    const char* version() override { return "0.1.0"; }

    uint32_t modalities() override { return UNIRT_MODALITY_LLM; }

    int32_t get_device_list(const unirt_GetDeviceListInput* input, unirt_GetDeviceListOutput* output) override {
        if (!input || !output) {
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        *output = {};
        if (!metal_available()) return UNIRT_SUCCESS;
        static std::string dev_name;
        if (dev_name.empty()) {
            if (metal_available()) {
                try {
                    const auto& info = mx::device_info(mx::Device(mx::Device::gpu));
                    auto        it   = info.find("device_name");
                    std::string base = (it != info.end() && std::holds_alternative<std::string>(it->second))
                                           ? std::get<std::string>(it->second)
                                           : "Apple Silicon GPU";
                    dev_name = base + " (MLX)";
                } catch (...) {
                    return UNIRT_SUCCESS;
                }
            }
        }
        auto ids   = (const char**)std::malloc(sizeof(const char*));
        auto names = (const char**)std::malloc(sizeof(const char*));
        if (!ids || !names) {
            std::free(ids);
            std::free(names);
            return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
        }
        ids[0]               = "mlx0";
        names[0]             = dev_name.c_str();
        output->device_ids   = ids;
        output->device_names = names;
        output->device_count = 1;
        return UNIRT_SUCCESS;
    }

    LlmBackend* create_llm() override {
        if (!metal_available()) {
            UNIRT_LOG_ERROR("mlx: cannot create a model without a usable Metal device");
            return nullptr;
        }
        try {
            mx::set_default_device(mx::Device(mx::Device::gpu));
            return new MlxLlm;
        } catch (const std::exception& e) {
            UNIRT_LOG_ERROR("mlx: failed to construct backend: {}", e.what());
            return nullptr;
        } catch (...) {
            return nullptr;
        }
    }
};

}  // namespace unirt::mlx_plugin

unirt_PluginId unirt_plugin_id() { return unirt::build_config::kPluginIdMlx; }

uint32_t unirt_plugin_abi_version() { return UNIRT_PLUGIN_ABI_VERSION; }

unirt_PluginTable* unirt_plugin_open() {
    UNIRT_LOG_TRACE("opening mlx plugin");
    return unirt::plugin_export::open_package<unirt::mlx_plugin::MlxPlugin>();
}

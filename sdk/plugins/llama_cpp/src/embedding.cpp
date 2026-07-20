// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "logging.h"

namespace unirt::llama_plugin {

namespace {

enum llama_pooling_type map_pooling(unirt_EmbeddingPooling requested) {
    switch (requested) {
        case UNIRT_EMBEDDING_POOLING_CLS:
            return LLAMA_POOLING_TYPE_CLS;
        case UNIRT_EMBEDDING_POOLING_MEAN:
            return LLAMA_POOLING_TYPE_MEAN;
        case UNIRT_EMBEDDING_POOLING_LAST_TOKEN:
            return LLAMA_POOLING_TYPE_LAST;
        case UNIRT_EMBEDDING_POOLING_MODEL_DEFAULT:
        default:
            return LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
}

/** RAII for llama_batch_init. */
struct BatchGuard {
    llama_batch batch;
    explicit BatchGuard(int32_t capacity) : batch(llama_batch_init(capacity, 0, 1)) {}
    ~BatchGuard() { llama_batch_free(batch); }
    BatchGuard(const BatchGuard&)            = delete;
    BatchGuard& operator=(const BatchGuard&) = delete;
};

}  // namespace

int32_t LlamaCppEmbedding::create(const unirt_EmbeddingCreateInput* input) {
    if (!input || !input->model_path || !input->model_path[0]) {
        UNIRT_LOG_ERROR("llama_cpp embedding: create input/model_path missing");
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    if (input->output_name && input->output_name[0]) {
        UNIRT_LOG_ERROR("llama_cpp embedding: output_name selects an ONNX graph output; GGUF has none");
        return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }

    llama_model_params model_params = llama_model_default_params();
    if (input->device_id && input->device_id[0]) {
        if (std::strcmp(input->device_id, "cpu") == 0) {
            model_params.n_gpu_layers = 0;
            device_label_             = "cpu";
        } else {
            ggml_backend_dev_t device = ggml_backend_dev_by_name(input->device_id);
            if (!device) {
                UNIRT_LOG_ERROR("llama_cpp embedding: device '{}' does not exist", input->device_id);
                return UNIRT_ERROR_COMMON_INVALID_DEVICE;
            }
            const char* description = ggml_backend_dev_description(device);
            device_label_           = description ? description : input->device_id;
        }
    } else {
        device_label_ = "default";
    }

    ModelPtr model(llama_model_load_from_file(input->model_path, model_params));
    if (!model) {
        UNIRT_LOG_ERROR("llama_cpp embedding: cannot load {}", input->model_path);
        return UNIRT_ERROR_COMMON_MODEL_LOAD;
    }
    if (llama_model_n_embd(model.get()) <= 0) {
        UNIRT_LOG_ERROR("llama_cpp embedding: model reports no embedding width");
        return UNIRT_ERROR_COMMON_MODEL_INVALID;
    }

    encoder_        = std::move(model);
    pooling_        = map_pooling(input->pooling);
    normalize_      = input->normalize;
    max_row_tokens_ = llama_model_n_ctx_train(encoder_.get());
    session_.reset();
    session_capacity_tokens_ = 0;
    session_capacity_seqs_   = 0;

    UNIRT_LOG_INFO(
        "llama_cpp embedding: loaded {} (width {}, max tokens {})", input->model_path,
        llama_model_n_embd(encoder_.get()), max_row_tokens_);
    return UNIRT_SUCCESS;
}

int32_t LlamaCppEmbedding::ensure_session(int32_t row_capacity, int32_t sequence_count) {
    if (session_ && session_capacity_tokens_ >= row_capacity && session_capacity_seqs_ >= sequence_count) {
        return UNIRT_SUCCESS;
    }
    // n_ctx is split evenly across n_seq_max inside llama.cpp, so keep it an
    // exact multiple; n_batch == n_ubatch == n_ctx mirrors the configuration
    // llama.cpp's own embedding tooling uses (non-causal attention needs the
    // whole batch in one ubatch).
    const auto span = static_cast<uint32_t>(row_capacity) * static_cast<uint32_t>(sequence_count);
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = span;
    params.n_batch              = span;
    params.n_ubatch             = span;
    params.n_seq_max            = static_cast<uint32_t>(sequence_count);
    params.embeddings           = true;
    params.pooling_type         = pooling_;

    session_.reset(llama_init_from_model(encoder_.get(), params));
    if (!session_) {
        UNIRT_LOG_ERROR(
            "llama_cpp embedding: cannot create a context for {} x {} tokens", sequence_count,
            row_capacity);
        session_capacity_tokens_ = 0;
        session_capacity_seqs_   = 0;
        return UNIRT_ERROR_EMBEDDING_INFERENCE_FAILED;
    }
    session_capacity_tokens_ = row_capacity;
    session_capacity_seqs_   = sequence_count;
    return UNIRT_SUCCESS;
}

int32_t LlamaCppEmbedding::encode(
    const unirt_EmbeddingEncodeInput* input, unirt_EmbeddingEncodeOutput* output) {
    if (output) *output = {};
    if (!encoder_) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;
    if (!input || !output || !input->input_ids || input->batch_size <= 0 ||
        input->sequence_length <= 0) {
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
    const int64_t cells =
        static_cast<int64_t>(input->batch_size) * static_cast<int64_t>(input->sequence_length);
    if (cells > std::numeric_limits<int32_t>::max()) return UNIRT_ERROR_COMMON_INVALID_INPUT;

    // Unpack rows: keep only unmasked tokens, validating ids against the
    // GGUF vocabulary (a mismatched tokenizer surfaces here, not as noise).
    const llama_vocab* vocabulary = llama_model_get_vocab(encoder_.get());
    const int32_t      vocab_size = llama_vocab_n_tokens(vocabulary);
    std::vector<std::vector<llama_token>> rows(static_cast<size_t>(input->batch_size));
    int32_t total_tokens = 0;
    for (int32_t row = 0; row < input->batch_size; ++row) {
        auto& tokens = rows[static_cast<size_t>(row)];
        for (int32_t column = 0; column < input->sequence_length; ++column) {
            const int64_t flat = static_cast<int64_t>(row) * input->sequence_length + column;
            if (input->attention_mask && input->attention_mask[flat] == 0) continue;
            const int64_t id = input->input_ids[flat];
            if (id < 0 || id >= vocab_size) {
                UNIRT_LOG_ERROR(
                    "llama_cpp embedding: token id {} outside vocabulary of {} (wrong tokenizer?)",
                    id, vocab_size);
                return UNIRT_ERROR_COMMON_INVALID_INPUT;
            }
            tokens.push_back(static_cast<llama_token>(id));
        }
        if (tokens.empty()) {
            UNIRT_LOG_ERROR("llama_cpp embedding: row {} has no unmasked tokens", row);
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        if (max_row_tokens_ > 0 && static_cast<int32_t>(tokens.size()) > max_row_tokens_) {
            UNIRT_LOG_ERROR(
                "llama_cpp embedding: row {} has {} tokens, model was trained for {}", row,
                tokens.size(), max_row_tokens_);
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
        }
        total_tokens += static_cast<int32_t>(tokens.size());
    }
    int32_t longest_row = 0;
    for (const auto& tokens : rows) {
        longest_row = std::max(longest_row, static_cast<int32_t>(tokens.size()));
    }

    const int32_t rc = ensure_session(longest_row, input->batch_size);
    if (rc != UNIRT_SUCCESS) return rc;
    llama_memory_clear(llama_get_memory(session_.get()), true);

    BatchGuard guard(total_tokens);
    llama_batch& batch = guard.batch;
    std::vector<int32_t> row_first_index(rows.size(), 0);
    for (size_t row = 0; row < rows.size(); ++row) {
        row_first_index[row] = batch.n_tokens;
        for (size_t position = 0; position < rows[row].size(); ++position) {
            const int32_t cursor  = batch.n_tokens++;
            batch.token[cursor]   = rows[row][position];
            batch.pos[cursor]     = static_cast<llama_pos>(position);
            batch.n_seq_id[cursor] = 1;
            batch.seq_id[cursor][0] = static_cast<llama_seq_id>(row);
            batch.logits[cursor]  = true;
        }
    }

    const bool encoder_only =
        llama_model_has_encoder(encoder_.get()) && !llama_model_has_decoder(encoder_.get());
    const int32_t status = encoder_only ? llama_encode(session_.get(), batch)
                                        : llama_decode(session_.get(), batch);
    if (status != 0) {
        UNIRT_LOG_ERROR("llama_cpp embedding: inference failed with status {}", status);
        return UNIRT_ERROR_EMBEDDING_INFERENCE_FAILED;
    }

    const int32_t width = llama_model_n_embd(encoder_.get());
    auto* result        = static_cast<float*>(
        std::malloc(sizeof(float) * static_cast<size_t>(input->batch_size) * width));
    if (!result) return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;

    const bool pooled = llama_pooling_type(session_.get()) != LLAMA_POOLING_TYPE_NONE;
    for (int32_t row = 0; row < input->batch_size; ++row) {
        float* destination = result + static_cast<size_t>(row) * width;
        if (pooled) {
            const float* vector = llama_get_embeddings_seq(session_.get(), row);
            if (!vector) {
                std::free(result);
                UNIRT_LOG_ERROR("llama_cpp embedding: model produced no pooled vector for row {}", row);
                return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
            }
            std::memcpy(destination, vector, sizeof(float) * static_cast<size_t>(width));
        } else {
            // The model declares no pooling: average its token-level output.
            const auto& tokens = rows[static_cast<size_t>(row)];
            std::fill(destination, destination + width, 0.0f);
            for (size_t position = 0; position < tokens.size(); ++position) {
                const float* token_vector = llama_get_embeddings_ith(
                    session_.get(), row_first_index[static_cast<size_t>(row)] +
                                        static_cast<int32_t>(position));
                if (!token_vector) {
                    std::free(result);
                    return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
                }
                for (int32_t axis = 0; axis < width; ++axis) destination[axis] += token_vector[axis];
            }
            const float scale = 1.0f / static_cast<float>(tokens.size());
            for (int32_t axis = 0; axis < width; ++axis) destination[axis] *= scale;
        }
        if (normalize_) {
            double squared = 0.0;
            for (int32_t axis = 0; axis < width; ++axis) {
                squared += static_cast<double>(destination[axis]) * destination[axis];
            }
            const double norm = std::sqrt(squared);
            if (norm > 0.0) {
                const auto scale = static_cast<float>(1.0 / norm);
                for (int32_t axis = 0; axis < width; ++axis) destination[axis] *= scale;
            }
        }
    }

    output->embeddings          = result;
    output->embedding_count     = input->batch_size;
    output->embedding_dimension = width;
    return UNIRT_SUCCESS;
}

int32_t LlamaCppEmbedding::get_runtime_stats(unirt_EmbeddingRuntimeStats* output) {
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *output = {};
    output->model_bytes       = -1;
    output->device_peak_bytes = -1;
    output->process_rss_bytes = -1;
    output->device_name       = device_label_.c_str();
    if (encoder_) output->model_bytes = static_cast<int64_t>(llama_model_size(encoder_.get()));
    return UNIRT_SUCCESS;
}

}  // namespace unirt::llama_plugin

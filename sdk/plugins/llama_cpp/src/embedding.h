// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

#include <llama.h>

#include "llm.h"  // ModelPtr / ContextPtr deleters
#include "plugin/embedding_backend.h"

namespace unirt::llama_plugin {

/**
 * Embedding encoder over GGUF models (BERT-family sentence transformers and
 * any other architecture llama.cpp can pool). Consumes the pre-tokenized
 * int64 batches of the embedding ABI: the caller's tokenizer must share the
 * vocabulary of the GGUF (true for any GGUF converted from the same
 * checkpoint as its tokenizer.json). Pooling runs inside llama.cpp; the
 * attention mask is honored by packing only unmasked tokens per sequence.
 */
class LlamaCppEmbedding final : public EmbeddingBackend {
   public:
    LlamaCppEmbedding() = default;
    ~LlamaCppEmbedding() override = default;

    LlamaCppEmbedding(const LlamaCppEmbedding&)            = delete;
    LlamaCppEmbedding& operator=(const LlamaCppEmbedding&) = delete;

    int32_t create(const unirt_EmbeddingCreateInput* input) override;
    int32_t encode(const unirt_EmbeddingEncodeInput* input, unirt_EmbeddingEncodeOutput* output) override;
    int32_t get_runtime_stats(unirt_EmbeddingRuntimeStats* output) override;

   private:
    /** (Re)build the llama_context when the current one is missing or too
     *  small for this batch shape. `row_capacity` is the longest sequence:
     *  llama.cpp divides n_ctx evenly across n_seq_max, so the context is
     *  sized row_capacity * sequence_count to stay divisible (a non-divisible
     *  n_ctx gets rounded down internally and breaks the reserve graph). */
    int32_t ensure_session(int32_t row_capacity, int32_t sequence_count);

    ModelPtr    encoder_;
    ContextPtr  session_;
    int32_t     session_capacity_tokens_ = 0;
    int32_t     session_capacity_seqs_   = 0;
    enum llama_pooling_type pooling_     = LLAMA_POOLING_TYPE_UNSPECIFIED;
    bool        normalize_               = false;
    int32_t     max_row_tokens_          = 0;  // model's trained context
    std::string device_label_            = "cpu";
};

}  // namespace unirt::llama_plugin

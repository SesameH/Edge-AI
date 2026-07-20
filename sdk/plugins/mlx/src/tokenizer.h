// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace unirt::mlx_plugin {

// Byte-level BPE tokenizer loaded from a HuggingFace tokenizer.json.
// Covers validated GPT-2/Llama-3-style byte-level layouts, including the
// Digits(individual_digits=true) + ByteLevel sequence used by SmolLM2.
class BpeTokenizer {
   public:
    // Throws std::runtime_error on parse failure.
    void load(const std::string& tokenizer_json_path);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string          decode(const std::vector<int32_t>& ids) const;
    // Decode of a single id, used for streaming. May hold back bytes that
    // are an incomplete UTF-8 sequence; caller receives them with the next id.
    std::string decode_piece(int32_t id) const;

    int32_t vocab_size() const { return static_cast<int32_t>(id_to_token_.size()); }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    bool    is_special(int32_t id) const { return special_ids_.count(id) > 0; }

   private:
    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::string>                 id_to_token_;
    // merge pair "left right" (space-joined) -> rank
    std::unordered_map<std::string, int32_t> merge_rank_;
    // special tokens (added_tokens): literal string -> id
    std::vector<std::pair<std::string, int32_t>> special_tokens_;
    std::unordered_map<int32_t, bool>            special_ids_;
    int32_t                                      bos_id_ = -1;
    int32_t                                      eos_id_ = -1;
    bool                                         individual_digits_ = false;

    std::vector<int32_t> bpe_segment(const std::string& byte_level_piece) const;
};

}  // namespace unirt::mlx_plugin

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

namespace unirt::mlx_plugin {

struct LlamaConfig {
    int32_t hidden_size          = 0;
    int32_t intermediate_size    = 0;
    int32_t num_hidden_layers    = 0;
    int32_t num_attention_heads  = 0;
    int32_t num_key_value_heads  = 0;
    int32_t vocab_size           = 0;
    int32_t max_position_embeddings = 0;
    float   rms_norm_eps         = 1e-5f;
    float   rope_theta           = 10000.f;
    bool    tie_word_embeddings  = true;
    int32_t bos_token_id         = -1;
    int32_t eos_token_id         = -1;
    // "quantization" block in config.json (mlx-community exports)
    bool    quantized            = false;
    int32_t quant_group_size     = 64;
    int32_t quant_bits           = 4;

    static LlamaConfig from_json_file(const std::string& path);
};

// A projection that is either dense (bf16/fp16) or MLX affine-quantized
// (packed uint32 weight + scales [+ biases]).
struct Linear {
    mlx::core::array                w{0.f};
    std::optional<mlx::core::array> scales;
    std::optional<mlx::core::array> biases;
    int32_t                         group_size = 64;
    int32_t                         bits       = 4;

    bool             is_quantized() const { return scales.has_value(); }
    mlx::core::array apply(const mlx::core::array& x) const;
    int64_t          nbytes() const;
};

// Minimal Llama-architecture decoder on MLX. Supports dense and
// MLX-quantized (4/8-bit affine) checkpoints; logits in float32.
class LlamaModel {
   public:
    void load(const std::string& model_dir, const LlamaConfig& cfg);

    // Feed `tokens`; append K/V to the cache; return float32 logits of the
    // last position (host copy, size vocab_size).
    std::vector<float> forward_logits(const std::vector<int32_t>& tokens);

    void    reset_cache();
    bool    trim_cache(int32_t n_past);
    int32_t n_past() const { return n_past_; }
    const LlamaConfig& config() const { return cfg_; }

    int64_t weights_bytes() const { return weights_bytes_; }
    int64_t kv_cache_bytes() const;

   private:
    struct Layer {
        Linear wq, wk, wv, wo;
        Linear w_gate, w_up, w_down;
        mlx::core::array input_norm{0.f}, post_attn_norm{0.f};
        std::optional<mlx::core::array> k_cache, v_cache;
    };

    LlamaConfig        cfg_;
    mlx::core::array   embed_{0.f};  // dense table for lookups (dequantized if needed)
    Linear             lm_head_;     // tied to embed weights; quantized form kept for logits
    mlx::core::array   final_norm_{0.f};
    std::vector<Layer> layers_;
    int32_t            n_past_        = 0;
    int64_t            weights_bytes_ = 0;
};

}  // namespace unirt::mlx_plugin

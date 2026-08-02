// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
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

// One checkpoint's parameters. Read-only once loaded, which is what lets
// several LlamaModel instances -- one per decoding slot -- share a single
// copy instead of each paying for its own.
struct LlamaWeights {
    struct Layer {
        Linear wq, wk, wv, wo;
        Linear w_gate, w_up, w_down;
        mlx::core::array input_norm{0.f}, post_attn_norm{0.f};
    };

    LlamaConfig        cfg;
    mlx::core::array   embed{0.f};  // dense table for lookups (dequantized if needed)
    Linear             lm_head;     // tied to embed weights; quantized form kept for logits
    mlx::core::array   final_norm{0.f};
    std::vector<Layer> layers;
    int64_t            bytes = 0;
};

using SharedWeights = std::shared_ptr<const LlamaWeights>;

// Read a checkpoint off disk. Every array is materialized before returning,
// so the result is safe to share across threads.
LlamaWeights load_weights(const std::string& model_dir, const LlamaConfig& cfg);

// The weights for `model_dir`, loaded once and shared from then on. A second
// slot on the same model costs a KV cache rather than a second copy of the
// parameters.
SharedWeights acquire_weights(const std::string& model_dir, const LlamaConfig& cfg);

// Minimal Llama-architecture decoder on MLX. Supports dense and
// MLX-quantized (4/8-bit affine) checkpoints; logits in float32.
//
// The parameters are shared (see LlamaWeights); the KV cache and the position
// it is filled to belong to this instance alone.
class LlamaModel {
   public:
    void load(const std::string& model_dir, const LlamaConfig& cfg);

    // Set the KV cache's hard ceiling (in tokens) and clear it. Must be
    // called once after load(), before the first forward_logits(), and not
    // changed afterwards. The cache's actual buffer starts small and grows
    // by doubling as needed (see forward_logits), capped at this ceiling —
    // it never allocates more than the conversation has actually used.
    void reserve_cache(int32_t max_capacity);

    // Feed `tokens`; append K/V to the cache; return float32 logits of the
    // last position (host copy, size vocab_size).
    std::vector<float> forward_logits(const std::vector<int32_t>& tokens);

    void    reset_cache();
    bool    trim_cache(int32_t n_past);
    int32_t n_past() const { return n_past_; }
    int32_t cache_capacity() const { return max_capacity_; }
    const LlamaConfig& config() const { return weights_->cfg; }

    // The size of the parameters this instance decodes with, whether or not
    // another slot is sharing them -- the same figure llama.cpp reports per
    // context.
    int64_t weights_bytes() const { return weights_ ? weights_->bytes : 0; }
    int64_t kv_cache_bytes() const;

   private:
    struct LayerCache {
        std::optional<mlx::core::array> k, v;
    };

    SharedWeights           weights_;
    std::vector<LayerCache> cache_;
    int32_t                 n_past_       = 0;
    int32_t                 max_capacity_ = 0;  // ceiling from reserve_cache (== n_ctx)
    int32_t                 allocated_    = 0;  // current physical buffer size, <= max_capacity_
};

}  // namespace unirt::mlx_plugin

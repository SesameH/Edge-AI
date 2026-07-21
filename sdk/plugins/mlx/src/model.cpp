// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "vendor/json.hpp"

namespace mx = mlx::core;

namespace unirt::mlx_plugin {

LlamaConfig LlamaConfig::from_json_file(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code size_error;
    const auto size = fs::file_size(path, size_error);
    if (size_error || size > 4 * 1024 * 1024) {
        throw std::runtime_error("config.json is missing or unreasonably large");
    }
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    nlohmann::json j;
    f >> j;
    if (j.value("model_type", "") != "llama") {
        throw std::runtime_error("mlx plugin currently supports model_type=llama only, got '" +
                                 j.value("model_type", "?") + "'");
    }
    LlamaConfig c;
    c.hidden_size             = j.at("hidden_size").get<int32_t>();
    c.intermediate_size       = j.at("intermediate_size").get<int32_t>();
    c.num_hidden_layers       = j.at("num_hidden_layers").get<int32_t>();
    c.num_attention_heads     = j.at("num_attention_heads").get<int32_t>();
    c.num_key_value_heads     = j.value("num_key_value_heads", c.num_attention_heads);
    c.vocab_size              = j.at("vocab_size").get<int32_t>();
    c.max_position_embeddings = j.value("max_position_embeddings", 4096);
    c.rms_norm_eps            = j.value("rms_norm_eps", 1e-5f);
    c.rope_theta              = j.value("rope_theta", 10000.f);
    c.tie_word_embeddings     = j.value("tie_word_embeddings", true);
    c.bos_token_id            = j.value("bos_token_id", -1);
    c.eos_token_id            = j.value("eos_token_id", -1);
    if (j.contains("quantization")) {
        c.quantized        = true;
        c.quant_group_size = j["quantization"].value("group_size", 64);
        c.quant_bits       = j["quantization"].value("bits", 4);
    }
    if (!c.tie_word_embeddings) {
        throw std::runtime_error("mlx plugin v1 requires tie_word_embeddings=true");
    }
    if (c.hidden_size <= 0 || c.intermediate_size <= 0 || c.num_hidden_layers <= 0 ||
        c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 || c.vocab_size <= 0 ||
        c.max_position_embeddings <= 0) {
        throw std::runtime_error("config contains non-positive model dimensions");
    }
    if (c.hidden_size % c.num_attention_heads != 0 ||
        c.num_attention_heads % c.num_key_value_heads != 0) {
        throw std::runtime_error("attention dimensions are inconsistent");
    }
    if (!std::isfinite(c.rms_norm_eps) || c.rms_norm_eps <= 0.f ||
        !std::isfinite(c.rope_theta) || c.rope_theta <= 0.f) {
        throw std::runtime_error("rms_norm_eps and rope_theta must be finite and positive");
    }
    if ((c.bos_token_id < -1 || c.bos_token_id >= c.vocab_size) ||
        (c.eos_token_id < -1 || c.eos_token_id >= c.vocab_size)) {
        throw std::runtime_error("BOS/EOS token id is outside the configured vocabulary");
    }
    if (c.quantized &&
        (c.quant_group_size <= 0 ||
         (c.quant_bits != 2 && c.quant_bits != 3 && c.quant_bits != 4 &&
          c.quant_bits != 6 && c.quant_bits != 8))) {
        throw std::runtime_error("unsupported MLX quantization group_size/bits");
    }
    return c;
}

mx::array Linear::apply(const mx::array& x) const {
    if (is_quantized()) {
        return mx::quantized_matmul(x, w, *scales, biases, /*transpose=*/true, group_size, bits);
    }
    return mx::matmul(x, mx::transpose(w));
}

int64_t Linear::nbytes() const {
    int64_t total = static_cast<int64_t>(w.nbytes());
    if (scales) total += static_cast<int64_t>(scales->nbytes());
    if (biases) total += static_cast<int64_t>(biases->nbytes());
    return total;
}

void LlamaModel::load(const std::string& model_dir, const LlamaConfig& cfg) {
    cfg_ = cfg;
    reset_cache();
    weights_bytes_ = 0;

    namespace fs = std::filesystem;
    const fs::path root = fs::path(model_dir);
    std::vector<fs::path> tensor_files;
    std::unordered_map<std::string, std::string> expected_weight_files;
    const fs::path index_path = root / "model.safetensors.index.json";
    if (fs::is_regular_file(index_path)) {
        std::error_code size_error;
        const auto index_size = fs::file_size(index_path, size_error);
        if (size_error || index_size > 64 * 1024 * 1024) {
            throw std::runtime_error("safetensors shard index is unreasonably large");
        }
        std::ifstream index_stream(index_path);
        nlohmann::json index;
        index_stream >> index;
        if (!index.contains("weight_map") || !index["weight_map"].is_object()) {
            throw std::runtime_error("invalid model.safetensors.index.json: missing weight_map");
        }
        std::unordered_set<std::string> seen;
        const fs::path canonical_root = fs::weakly_canonical(root);
        for (auto it = index["weight_map"].begin(); it != index["weight_map"].end(); ++it) {
            const std::string relative = it.value().get<std::string>();
            fs::path rel(relative);
            if (rel.empty() || rel.is_absolute() ||
                std::find(rel.begin(), rel.end(), fs::path("..")) != rel.end() ||
                rel.extension() != ".safetensors") {
                throw std::runtime_error("unsafe shard path in safetensors index: " + relative);
            }
            const fs::path candidate = fs::weakly_canonical(root / rel);
            const fs::path within_root = candidate.lexically_relative(canonical_root);
            if (within_root.empty() ||
                (!within_root.empty() && *within_root.begin() == fs::path(".."))) {
                throw std::runtime_error("safetensors shard resolves outside the model directory: " + relative);
            }
            expected_weight_files.emplace(it.key(), rel.generic_string());
            if (seen.insert(relative).second) tensor_files.push_back(root / rel);
        }
    } else if (fs::is_regular_file(root / "model.safetensors")) {
        tensor_files.push_back(root / "model.safetensors");
    } else {
        const std::regex shard_pattern(R"(^model-([0-9]+)-of-([0-9]+)\.safetensors$)");
        std::set<int32_t> shard_indexes;
        int32_t shard_total = -1;
        for (const auto& entry : fs::directory_iterator(root)) {
            std::smatch match;
            const std::string name = entry.path().filename().string();
            if (entry.is_regular_file() && std::regex_match(name, match, shard_pattern)) {
                const int32_t index = std::stoi(match[1].str());
                const int32_t total = std::stoi(match[2].str());
                if (shard_total != -1 && shard_total != total) {
                    throw std::runtime_error("safetensors shards disagree on total count");
                }
                shard_total = total;
                shard_indexes.insert(index);
                tensor_files.push_back(entry.path());
            }
        }
        if (shard_total > 0 &&
            (static_cast<int32_t>(shard_indexes.size()) != shard_total ||
             *shard_indexes.begin() != 1 || *shard_indexes.rbegin() != shard_total)) {
            throw std::runtime_error("safetensors shard set is incomplete");
        }
    }
    std::sort(tensor_files.begin(), tensor_files.end());
    if (tensor_files.empty()) {
        throw std::runtime_error("no model safetensors file or shard index found");
    }

    std::unordered_map<std::string, mx::array> weights;
    std::unordered_set<std::string> indexed_weights_seen;
    for (const auto& tensor_file : tensor_files) {
        if (!fs::is_regular_file(tensor_file)) {
            throw std::runtime_error("missing safetensors shard: " + tensor_file.string());
        }
        auto [shard, metadata] = mx::load_safetensors(tensor_file.string());
        for (auto& [name, value] : shard) {
            if (!expected_weight_files.empty()) {
                const auto expected = expected_weight_files.find(name);
                const std::string relative = fs::relative(tensor_file, root).generic_string();
                if (expected == expected_weight_files.end() || expected->second != relative) {
                    throw std::runtime_error("safetensors tensor does not match its shard index: " + name);
                }
                indexed_weights_seen.insert(name);
            }
            if (!weights.emplace(name, std::move(value)).second) {
                throw std::runtime_error("duplicate tensor across safetensors shards: " + name);
            }
        }
    }
    if (!expected_weight_files.empty() && indexed_weights_seen.size() != expected_weight_files.size()) {
        throw std::runtime_error("safetensors shard index references missing tensors");
    }

    auto get = [&](const std::string& name) -> mx::array {
        auto it = weights.find(name);
        if (it == weights.end()) throw std::runtime_error("missing weight: " + name);
        return it->second;
    };
    auto make_linear = [&](const std::string& name) -> Linear {
        Linear l;
        l.w = get(name + ".weight");
        auto s = weights.find(name + ".scales");
        if (s != weights.end()) {
            l.scales     = s->second;
            l.group_size = cfg.quant_group_size;
            l.bits       = cfg.quant_bits;
            auto b       = weights.find(name + ".biases");
            if (b != weights.end()) l.biases = b->second;
        }
        return l;
    };
    auto require_shape = [](const mx::array& value, const std::string& name,
                            std::initializer_list<int32_t> expected) {
        if (value.ndim() != expected.size() ||
            !std::equal(value.shape().begin(), value.shape().end(), expected.begin())) {
            throw std::runtime_error("unexpected shape for tensor: " + name);
        }
    };
    auto validate_linear = [&](const Linear& linear, const std::string& name,
                               int32_t output_size, int32_t input_size) {
        if (linear.w.ndim() != 2 || linear.w.shape(0) != output_size) {
            throw std::runtime_error("unexpected shape for tensor: " + name + ".weight");
        }
        if (!linear.is_quantized()) {
            if (linear.w.shape(1) != input_size) {
                throw std::runtime_error("unexpected shape for tensor: " + name + ".weight");
            }
            return;
        }
        if (!linear.scales || linear.scales->ndim() != 2 ||
            linear.scales->shape(0) != output_size ||
            linear.w.shape(1) * 32 < input_size * linear.bits) {
            throw std::runtime_error("invalid quantized tensors for: " + name);
        }
    };

    lm_head_ = make_linear("model.embed_tokens");
    // Embedding lookups need a dense table; dequantize once if necessary.
    embed_ = lm_head_.is_quantized()
                 ? mx::dequantize(lm_head_.w, *lm_head_.scales, lm_head_.biases,
                                  lm_head_.group_size, lm_head_.bits)
                 : lm_head_.w;
    require_shape(embed_, "model.embed_tokens.weight", {cfg.vocab_size, cfg.hidden_size});
    final_norm_ = get("model.norm.weight");
    require_shape(final_norm_, "model.norm.weight", {cfg.hidden_size});

    layers_.clear();
    layers_.resize(cfg.num_hidden_layers);
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        Layer&            l = layers_[static_cast<size_t>(i)];
        l.wq             = make_linear(p + "self_attn.q_proj");
        l.wk             = make_linear(p + "self_attn.k_proj");
        l.wv             = make_linear(p + "self_attn.v_proj");
        l.wo             = make_linear(p + "self_attn.o_proj");
        l.w_gate         = make_linear(p + "mlp.gate_proj");
        l.w_up           = make_linear(p + "mlp.up_proj");
        l.w_down         = make_linear(p + "mlp.down_proj");
        l.input_norm     = get(p + "input_layernorm.weight");
        l.post_attn_norm = get(p + "post_attention_layernorm.weight");
        const int32_t head_dim = cfg.hidden_size / cfg.num_attention_heads;
        validate_linear(l.wq, p + "self_attn.q_proj", cfg.hidden_size, cfg.hidden_size);
        validate_linear(l.wk, p + "self_attn.k_proj", cfg.num_key_value_heads * head_dim, cfg.hidden_size);
        validate_linear(l.wv, p + "self_attn.v_proj", cfg.num_key_value_heads * head_dim, cfg.hidden_size);
        validate_linear(l.wo, p + "self_attn.o_proj", cfg.hidden_size, cfg.hidden_size);
        validate_linear(l.w_gate, p + "mlp.gate_proj", cfg.intermediate_size, cfg.hidden_size);
        validate_linear(l.w_up, p + "mlp.up_proj", cfg.intermediate_size, cfg.hidden_size);
        validate_linear(l.w_down, p + "mlp.down_proj", cfg.hidden_size, cfg.intermediate_size);
        require_shape(l.input_norm, p + "input_layernorm.weight", {cfg.hidden_size});
        require_shape(l.post_attn_norm, p + "post_attention_layernorm.weight", {cfg.hidden_size});
    }
    n_past_ = 0;

    // Materialize all weights now so create() pays the load cost, not the
    // first generate().
    std::vector<mx::array> all;
    for (auto& [k, v] : weights) all.push_back(v);
    all.push_back(embed_);
    mx::eval(all);

    weights_bytes_ = static_cast<int64_t>(embed_.nbytes());
    if (lm_head_.is_quantized()) weights_bytes_ += lm_head_.nbytes();
    weights_bytes_ += static_cast<int64_t>(final_norm_.nbytes());
    for (auto& l : layers_) {
        weights_bytes_ += l.wq.nbytes() + l.wk.nbytes() + l.wv.nbytes() + l.wo.nbytes() +
                          l.w_gate.nbytes() + l.w_up.nbytes() + l.w_down.nbytes() +
                          static_cast<int64_t>(l.input_norm.nbytes()) +
                          static_cast<int64_t>(l.post_attn_norm.nbytes());
    }
}

namespace {
constexpr int32_t kCacheInitialCapacity = 256;
}  // namespace

void LlamaModel::reserve_cache(int32_t max_capacity) {
    if (max_capacity <= 0) throw std::invalid_argument("cache capacity must be positive");
    max_capacity_ = max_capacity;
    allocated_    = 0;
    for (auto& l : layers_) {
        l.k_cache.reset();
        l.v_cache.reset();
    }
    n_past_ = 0;
}

int64_t LlamaModel::kv_cache_bytes() const {
    // The buffer only ever holds allocated_ tokens' worth of physical
    // storage (not max_capacity_'s), but nbytes() alone would still report
    // the fully-allocated size rather than how much of it holds live KV
    // data for the current transcript; scale by the in-use fraction.
    if (allocated_ <= 0) return 0;
    int64_t total = 0;
    for (const auto& l : layers_) {
        if (l.k_cache) total += static_cast<int64_t>(l.k_cache->nbytes()) * n_past_ / allocated_;
        if (l.v_cache) total += static_cast<int64_t>(l.v_cache->nbytes()) * n_past_ / allocated_;
    }
    return total;
}

std::vector<float> LlamaModel::forward_logits(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) throw std::invalid_argument("forward_logits requires at least one token");
    for (int32_t token : tokens) {
        if (token < 0 || token >= cfg_.vocab_size) {
            throw std::out_of_range("token id is outside the model vocabulary");
        }
    }
    if (max_capacity_ <= 0) {
        throw std::runtime_error("forward_logits called before reserve_cache");
    }
    const int32_t T        = static_cast<int32_t>(tokens.size());
    if (n_past_ + T > max_capacity_) {
        throw std::out_of_range("forward_logits would exceed the reserved KV cache capacity");
    }
    const int32_t n_heads  = cfg_.num_attention_heads;
    const int32_t n_kv     = cfg_.num_key_value_heads;
    const int32_t head_dim = cfg_.hidden_size / n_heads;

    // Grow the physical buffer by doubling (capped at max_capacity_) rather
    // than allocating the whole ceiling up front — most steps just write
    // into already-reserved slots; only the occasional doubling event
    // pays a one-time copy-forward of the valid prefix.
    int32_t new_capacity = allocated_;
    if (n_past_ + T > new_capacity) {
        new_capacity = new_capacity > 0 ? new_capacity : kCacheInitialCapacity;
        while (new_capacity < n_past_ + T) new_capacity *= 2;
        new_capacity = std::min(new_capacity, max_capacity_);
    }

    mx::array ids = mx::array(tokens.data(), {1, T}, mx::int32);
    mx::array x   = mx::take(embed_, ids, 0);  // [1, T, H]

    for (auto& l : layers_) {
        // ---- attention ----
        mx::array h = mx::fast::rms_norm(x, l.input_norm, cfg_.rms_norm_eps);

        mx::array q = l.wq.apply(h);
        mx::array k = l.wk.apply(h);
        mx::array v = l.wv.apply(h);

        q = mx::transpose(mx::reshape(q, {1, T, n_heads, head_dim}), {0, 2, 1, 3});
        k = mx::transpose(mx::reshape(k, {1, T, n_kv, head_dim}), {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v, {1, T, n_kv, head_dim}), {0, 2, 1, 3});

        q = mx::fast::rope(q, head_dim, /*traditional=*/false, cfg_.rope_theta, /*scale=*/1.f, n_past_);
        k = mx::fast::rope(k, head_dim, /*traditional=*/false, cfg_.rope_theta, /*scale=*/1.f, n_past_);

        if (!l.k_cache) {
            // First write since load()/reserve_cache()/reset_cache().
            const mx::Shape full_shape{1, n_kv, new_capacity, head_dim};
            l.k_cache = mx::zeros(full_shape, k.dtype());
            l.v_cache = mx::zeros(full_shape, v.dtype());
        } else if (new_capacity > allocated_) {
            // Doubling event: reallocate at the new size and copy forward
            // only the valid prefix (garbage past n_past_ is never read).
            const mx::Shape full_shape{1, n_kv, new_capacity, head_dim};
            mx::array grown_k = mx::zeros(full_shape, l.k_cache->dtype());
            mx::array grown_v = mx::zeros(full_shape, l.v_cache->dtype());
            if (n_past_ > 0) {
                mx::array old_k = mx::slice(*l.k_cache, {0, 0, 0, 0}, {1, n_kv, n_past_, head_dim});
                mx::array old_v = mx::slice(*l.v_cache, {0, 0, 0, 0}, {1, n_kv, n_past_, head_dim});
                grown_k = mx::slice_update(grown_k, old_k, {0, 0, 0, 0}, {1, n_kv, n_past_, head_dim});
                grown_v = mx::slice_update(grown_v, old_v, {0, 0, 0, 0}, {1, n_kv, n_past_, head_dim});
            }
            l.k_cache = grown_k;
            l.v_cache = grown_v;
        }
        l.k_cache = mx::slice_update(*l.k_cache, k, {0, 0, n_past_, 0},
                                      {1, n_kv, n_past_ + T, head_dim});
        l.v_cache = mx::slice_update(*l.v_cache, v, {0, 0, n_past_, 0},
                                      {1, n_kv, n_past_ + T, head_dim});
        mx::array k_valid = mx::slice(*l.k_cache, {0, 0, 0, 0}, {1, n_kv, n_past_ + T, head_dim});
        mx::array v_valid = mx::slice(*l.v_cache, {0, 0, 0, 0}, {1, n_kv, n_past_ + T, head_dim});

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        mx::array   attn  = (T > 1)
                                ? mx::fast::scaled_dot_product_attention(q, k_valid, v_valid, scale, "causal", std::nullopt)
                                : mx::fast::scaled_dot_product_attention(q, k_valid, v_valid, scale, "", std::nullopt);

        attn = mx::reshape(mx::transpose(attn, {0, 2, 1, 3}), {1, T, n_heads * head_dim});
        x    = x + l.wo.apply(attn);

        // ---- mlp ----
        h              = mx::fast::rms_norm(x, l.post_attn_norm, cfg_.rms_norm_eps);
        mx::array gate = l.w_gate.apply(h);
        mx::array up   = l.w_up.apply(h);
        x              = x + l.w_down.apply(mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), up));
    }

    x = mx::fast::rms_norm(x, final_norm_, cfg_.rms_norm_eps);
    // last position only; tied embeddings → logits via the (possibly
    // quantized) embedding projection
    mx::array last   = mx::slice(x, {0, T - 1, 0}, {1, T, cfg_.hidden_size}, {1, 1, 1});
    mx::array logits = mx::astype(mx::reshape(lm_head_.apply(last), {cfg_.vocab_size}), mx::float32);
    mx::eval(logits);

    n_past_    += T;
    allocated_  = new_capacity;

    std::vector<float> out(static_cast<size_t>(cfg_.vocab_size));
    std::memcpy(out.data(), logits.data<float>(), out.size() * sizeof(float));
    return out;
}

void LlamaModel::reset_cache() {
    // Buffers stay allocated at their reserved capacity; only the valid-
    // length marker resets. The next forward_logits() overwrites from
    // slot 0, so there's nothing to free or re-zero here.
    n_past_ = 0;
}

bool LlamaModel::trim_cache(int32_t n_past) {
    if (n_past < 0 || n_past > n_past_) return false;
    // Rewinding a fixed-capacity buffer is just moving the valid-length
    // marker back — slots past the new n_past are never read (every
    // forward_logits() call slices [0, n_past_+T)) and get overwritten in
    // place once new tokens land there.
    n_past_ = n_past;
    return true;
}

}  // namespace unirt::mlx_plugin

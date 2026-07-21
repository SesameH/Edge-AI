// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include "unirt.h"

namespace unirt::mlx_plugin {

// CPU-side sampler over a host logits vector. temperature <= 0 → greedy.
class Sampler {
   public:
    explicit Sampler(const unirt_SamplerConfig* cfg) {
        if (cfg) {
            temperature_ = cfg->temperature;
            top_p_       = cfg->top_p;
            top_k_       = cfg->top_k;
            min_p_       = cfg->min_p;
            rep_pen_     = cfg->repetition_penalty;
            presence_    = cfg->presence_penalty;
            frequency_   = cfg->frequency_penalty;
            seed_        = cfg->seed;
        }
        rng_.seed(seed_ > 0 ? static_cast<uint32_t>(seed_) : std::random_device{}());
    }

    void observe(int32_t token) { ++counts_[token]; }

    int32_t sample(std::vector<float>& logits) {
        if (logits.empty()) return -1;
        apply_penalties(logits);

        if (temperature_ <= 0.f) {
            return static_cast<int32_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
        }

        for (auto& l : logits) l /= temperature_;

        // candidate set: top-k (0 = all). idx_/probs_ are member scratch
        // reused across every token in a generate() call instead of a
        // fresh heap allocation per step.
        idx_.resize(logits.size());
        std::iota(idx_.begin(), idx_.end(), 0);
        size_t k = (top_k_ > 0 && static_cast<size_t>(top_k_) < idx_.size())
                       ? static_cast<size_t>(top_k_)
                       : idx_.size();
        std::partial_sort(idx_.begin(), idx_.begin() + k, idx_.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        idx_.resize(k);

        // softmax over candidates
        float max_l = logits[idx_[0]];
        probs_.resize(k);
        float sum = 0.f;
        for (size_t i = 0; i < k; ++i) {
            probs_[i] = std::exp(logits[idx_[i]] - max_l);
            sum += probs_[i];
        }
        for (auto& p : probs_) p /= sum;

        // min-p: drop candidates below min_p * p_max
        if (min_p_ > 0.f) {
            float cutoff = min_p_ * probs_[0];
            size_t end   = k;
            while (end > 1 && probs_[end - 1] < cutoff) --end;
            probs_.resize(end);
            idx_.resize(end);
        }

        // top-p: smallest prefix with cumulative prob >= top_p
        if (top_p_ > 0.f && top_p_ < 1.f) {
            float  cum = 0.f;
            size_t end = probs_.size();
            for (size_t i = 0; i < probs_.size(); ++i) {
                cum += probs_[i];
                if (cum >= top_p_) {
                    end = i + 1;
                    break;
                }
            }
            probs_.resize(end);
            idx_.resize(end);
        }

        std::discrete_distribution<size_t> dist(probs_.begin(), probs_.end());
        return idx_[dist(rng_)];
    }

   private:
    void apply_penalties(std::vector<float>& logits) {
        if (rep_pen_ <= 0.f && presence_ == 0.f && frequency_ == 0.f) return;
        for (const auto& [tok, count] : counts_) {
            if (tok < 0 || static_cast<size_t>(tok) >= logits.size()) continue;
            float& l = logits[static_cast<size_t>(tok)];
            if (rep_pen_ > 0.f && rep_pen_ != 1.f) {
                l = l > 0 ? l / rep_pen_ : l * rep_pen_;
            }
            l -= presence_ + frequency_ * static_cast<float>(count);
        }
    }

    float   temperature_ = 0.f;
    float   top_p_       = 0.f;
    int32_t top_k_       = 0;
    float   min_p_       = 0.f;
    float   rep_pen_     = 0.f;
    float   presence_    = 0.f;
    float   frequency_   = 0.f;
    int32_t seed_        = 0;

    std::mt19937                       rng_;
    std::unordered_map<int32_t, int>   counts_;
    std::vector<int32_t>               idx_;
    std::vector<float>                 probs_;
};

}  // namespace unirt::mlx_plugin

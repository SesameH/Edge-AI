// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "unirt.h"

namespace unirt {

/** Derive the rate fields of a profile from the raw counters the plugin
 *  filled in. Times are microseconds; rates come out in tokens/second. */
inline void finalize_profile(unirt_ProfileData& profile) noexcept {
    constexpr double kUsPerSecond = 1e6;
    profile.prefill_speed =
        profile.prompt_time > 0
            ? static_cast<double>(profile.prompt_tokens) * kUsPerSecond / static_cast<double>(profile.prompt_time)
            : 0.0;
    profile.decoding_speed =
        profile.decode_time > 0
            ? static_cast<double>(profile.generated_tokens) * kUsPerSecond / static_cast<double>(profile.decode_time)
            : 0.0;
    profile.real_time_factor =
        profile.audio_duration > 0
            ? static_cast<double>(profile.decode_time) / static_cast<double>(profile.audio_duration)
            : 0.0;
}

}  // namespace unirt

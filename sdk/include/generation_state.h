// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "unirt.h"

namespace unirt {

// Owns the user-visible text during one generation.  It retains any suffix
// that could still become a stop sequence, preventing stop bytes from leaking
// through the streaming callback before the match is complete.
class StopStreamState {
   public:
    explicit StopStreamState(std::vector<std::string> stops) : stops_(std::move(stops)) {}

    void append(const std::string& piece) { text_ += piece; }

    bool find_and_trim_stop() {
        size_t first = std::string::npos;
        for (const auto& stop : stops_) {
            if (stop.empty()) continue;
            const size_t position = text_.find(stop);
            if (position != std::string::npos) first = std::min(first, position);
        }
        if (first == std::string::npos) return false;
        text_.resize(first);
        return true;
    }

    // Returns false when the callback requests cancellation.  The callback
    // has consumed the entire delivered range before returning false.
    bool emit_safe(unirt_token_callback callback, void* user_data, bool final = false) {
        if (!callback) {
            emitted_ = text_.size();
            return true;
        }
        size_t safe_end = text_.size();
        if (!final) safe_end -= pending_stop_prefix();
        if (safe_end <= emitted_) return true;
        const std::string chunk = text_.substr(emitted_, safe_end - emitted_);
        emitted_ = safe_end;
        return callback(chunk.c_str(), user_data);
    }

    // On user cancellation, bytes retained for stop detection were never
    // observed by the caller and therefore must not appear in full_text.
    void discard_unemitted() { text_.resize(emitted_); }

    // Byte-level tokenizers can stop between UTF-8 code units at max_tokens.
    // Remove only an unfinished trailing character so the C ABI never returns
    // malformed UTF-8 in full_text.
    void discard_incomplete_utf8_tail() {
        if (text_.empty()) return;
        size_t start = text_.size() - 1;
        while (start > 0 &&
               (static_cast<unsigned char>(text_[start]) & 0xC0) == 0x80) {
            --start;
        }
        const unsigned char lead = static_cast<unsigned char>(text_[start]);
        size_t expected = 1;
        if ((lead & 0xE0) == 0xC0) expected = 2;
        else if ((lead & 0xF0) == 0xE0) expected = 3;
        else if ((lead & 0xF8) == 0xF0) expected = 4;
        if (text_.size() - start < expected) text_.resize(start);
    }

    const std::string& text() const { return text_; }

   private:
    size_t pending_stop_prefix() const {
        size_t pending = 0;
        for (const auto& stop : stops_) {
            if (stop.empty()) continue;
            const size_t limit = std::min(text_.size(), stop.size() - 1);
            for (size_t length = limit; length > pending; --length) {
                if (text_.compare(text_.size() - length, length, stop, 0, length) == 0) {
                    pending = length;
                    break;
                }
            }
        }
        return pending;
    }

    std::vector<std::string> stops_;
    std::string              text_;
    size_t                   emitted_ = 0;
};

}  // namespace unirt

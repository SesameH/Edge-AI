// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <filesystem>
#include <string>

#include "unirt.h"

namespace unirt {

/** Directory containing the shared library this code is linked into.
 *  Used as the default plugin search root. Throws on failure. */
std::filesystem::path module_directory();

/** Thread-local failure detail behind unirt_last_error_message(). The
 *  bridge clears it on entry to each C entrypoint and records prose when a
 *  failure has more to say than its error code. */
void set_last_error(std::string message) noexcept;
void clear_last_error() noexcept;
const char* last_error_cstr() noexcept;

/**
 * Reassembles a stream of byte pieces into valid UTF-8 before handing it to
 * a caller-supplied token callback. Plugins emit raw token pieces, which may
 * end mid-codepoint; the joiner holds the incomplete tail until the bytes
 * that finish it arrive.
 *
 * Usage: construct with the caller's callback, point the plugin at
 * `trampoline()` with the instance as user_data, call `finish()` after a
 * successful generation to flush any buffered tail.
 */
class StreamJoiner {
   public:
    StreamJoiner(unirt_token_callback downstream, void* downstream_data) noexcept
        : downstream_(downstream), downstream_data_(downstream_data) {}

    /** The C-compatible callback to hand to the plugin. `user_data` must be
     *  the StreamJoiner instance. */
    static unirt_token_callback trampoline() noexcept;

    /** Deliver any buffered (possibly incomplete) tail downstream verbatim.
     *  Call once, after generation reports success. */
    void finish();

   private:
    bool consume(const char* piece);
    /** Split the buffer at the last complete codepoint boundary; returns the
     *  deliverable part and keeps the incomplete tail buffered. */
    std::string take_complete_prefix();

    unirt_token_callback downstream_;
    void*                downstream_data_;
    std::string          pending_;
    bool                 downstream_stopped_ = false;
};

}  // namespace unirt

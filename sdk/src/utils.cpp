// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "utils.h"

#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
extern "C" IMAGE_DOS_HEADER __ImageBase;
#else
#include <dlfcn.h>
#endif

namespace unirt {

std::filesystem::path module_directory() {
#if defined(_WIN32)
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(reinterpret_cast<HMODULE>(&__ImageBase), path, MAX_PATH);
    return std::filesystem::path(path).parent_path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&module_directory), &info) && info.dli_fname) {
        return std::filesystem::canonical(info.dli_fname).parent_path();
    }
    throw std::runtime_error("cannot locate the unirt shared library on disk");
#endif
}

namespace {

/** Bytes a UTF-8 sequence starting with `lead` should occupy; 1 for ASCII
 *  and for invalid lead bytes (so they pass through instead of wedging). */
size_t sequence_length(unsigned char lead) noexcept {
    if ((lead & 0x80) == 0x00) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;
}

bool is_continuation(unsigned char byte) noexcept { return (byte & 0xC0) == 0x80; }

}  // namespace

std::string StreamJoiner::take_complete_prefix() {
    if (pending_.empty()) return {};

    // Walk back to the lead byte of the final codepoint.
    size_t lead_at = pending_.size();
    while (lead_at > 0) {
        --lead_at;
        if (!is_continuation(static_cast<unsigned char>(pending_[lead_at]))) break;
    }

    const size_t have = pending_.size() - lead_at;
    const size_t want = sequence_length(static_cast<unsigned char>(pending_[lead_at]));

    std::string ready;
    if (have >= want) {
        ready = std::move(pending_);
        pending_.clear();
    } else {
        ready = pending_.substr(0, lead_at);
        pending_.erase(0, lead_at);
    }
    return ready;
}

bool StreamJoiner::consume(const char* piece) {
    if (!piece) return false;
    pending_.append(piece);
    const std::string ready = take_complete_prefix();
    if (!ready.empty() && downstream_) {
        const bool keep_going = downstream_(ready.c_str(), downstream_data_);
        downstream_stopped_   = !keep_going;
        return keep_going;
    }
    return !downstream_stopped_;
}

void StreamJoiner::finish() {
    if (pending_.empty() || !downstream_ || downstream_stopped_) return;
    downstream_(pending_.c_str(), downstream_data_);
    pending_.clear();
}

unirt_token_callback StreamJoiner::trampoline() noexcept {
    return [](const char* piece, void* self) -> bool {
        return self ? static_cast<StreamJoiner*>(self)->consume(piece) : false;
    };
}

}  // namespace unirt

namespace unirt {

namespace {
thread_local std::string t_last_error;
}

void set_last_error(std::string message) noexcept {
    try {
        t_last_error = std::move(message);
    } catch (...) {
        // Even recording the failure failed; leave whatever was there.
    }
}

void clear_last_error() noexcept {
    t_last_error.clear();
}

const char* last_error_cstr() noexcept { return t_last_error.c_str(); }

}  // namespace unirt

const char* unirt_last_error_message(void) { return unirt::last_error_cstr(); }

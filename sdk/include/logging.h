// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

/**
 * Logging for the bridge and plugins.
 *
 * Records flow through the process-wide `unirt_log` callback (settable via
 * unirt_set_log). Formatting uses the bundled {fmt}; every record carries a
 * `[file:line:function]` prefix. All levels are forwarded — filtering is the
 * log consumer's job, so `unirt_log_level` stays at TRACE by default.
 */

#define FMT_HEADER_ONLY
#ifndef FMT_USE_CONSTEVAL
#define FMT_USE_CONSTEVAL 0
#endif

#include <atomic>
#include <cstring>
#include <type_traits>

#include "external/fmt/core.h"
#include "unirt.h"

UNIRT_API extern std::atomic<unirt_log_callback> unirt_log;
UNIRT_API extern unirt_LogLevel                   unirt_log_level;

namespace unirt::logging {

/** Render one argument. C strings print their text; every other pointer
 *  prints as an address, so struct pointers in trace logs are cheap and can
 *  never dereference null.
 *
 *  Because every argument becomes a string before the pattern is applied,
 *  log patterns take plain `{}` only. A numeric spec like `{:.1f}` compiles
 *  -- fmt checks it against the argument's original type -- and then throws
 *  "invalid format specifier" at runtime, when the argument it actually gets
 *  is a string. Format the value yourself if you need a particular shape. */
template <typename Arg>
inline auto printable(Arg value) {
    if constexpr (std::is_pointer_v<Arg>) {
        using Pointee = std::remove_cv_t<std::remove_pointer_t<Arg>>;
        if (value == nullptr) return fmt::format("nullptr");
        if constexpr (std::is_same_v<Pointee, char>) {
            return fmt::format("{}", value);
        } else {
            return fmt::format("{}", static_cast<const void*>(value));
        }
    } else {
        return fmt::format("{}", value);
    }
}

/** Strip the build-tree prefix so log locations are repo-relative. */
inline const char* trim_source_path(const char* file) {
    const char* anchored = std::strstr(file, PROJECT_SOURCE_DIR);
    return anchored ? anchored + std::strlen(PROJECT_SOURCE_DIR) + 1 : file;
}

template <typename... Args>
void emit(
    unirt_LogLevel level, const char* file, int32_t line, const char* function,
    fmt::format_string<Args...> pattern, Args&&... args) {
    const auto sink = unirt_log.load(std::memory_order_acquire);
    if (!sink) return;
    sink(level,
         fmt::format(
             "[{}:{}:{}] {}", trim_source_path(file), line, function,
             fmt::format(pattern, printable(std::forward<Args>(args))...))
             .c_str());
}

}  // namespace unirt::logging

#define UNIRT_LEVEL_LOG(level, ...)                                                        \
    do {                                                                                    \
        if ((level) >= unirt_log_level) {                                                  \
            ::unirt::logging::emit((level), __FILE__, __LINE__, __func__, __VA_ARGS__);    \
        }                                                                                   \
    } while (0)

#define UNIRT_LOG_TRACE(...) UNIRT_LEVEL_LOG(UNIRT_LOG_LEVEL_TRACE, __VA_ARGS__)
#define UNIRT_LOG_DEBUG(...) UNIRT_LEVEL_LOG(UNIRT_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define UNIRT_LOG_INFO(...) UNIRT_LEVEL_LOG(UNIRT_LOG_LEVEL_INFO, __VA_ARGS__)
#define UNIRT_LOG_WARN(...) UNIRT_LEVEL_LOG(UNIRT_LOG_LEVEL_WARN, __VA_ARGS__)
#define UNIRT_LOG_ERROR(...) UNIRT_LEVEL_LOG(UNIRT_LOG_LEVEL_ERROR, __VA_ARGS__)

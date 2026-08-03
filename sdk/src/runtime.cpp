// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// Runtime lifecycle (init / deinit), logging sink, and the plugin/device
// enumeration entrypoints.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

#include "unirt.h"
#include "bridge_support.h"
#include "build_config.h"
#include "handle_registry.h"
#include "logging.h"
#include "registry.h"

#ifdef _WIN32
#include <windows.h>
#endif

// Baseline armv8.0 boards lack the armv8.2 features this build bakes in;
// refuse to start instead of dying with SIGILL deep inside a backend.
#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>

static bool host_cpu_is_capable() {
    const unsigned long required =
        HWCAP_ATOMICS | HWCAP_ASIMDRDM | HWCAP_ASIMDDP | HWCAP_FPHP | HWCAP_ASIMDHP | HWCAP_CRC32;
    return (getauxval(AT_HWCAP) & required) == required;
}
#else
static bool host_cpu_is_capable() { return true; }
#endif

using namespace unirt;

namespace {

struct Lifecycle {
    std::mutex mutex;
    bool       running = false;
};

Lifecycle& lifecycle() {
    static Lifecycle state;
    return state;
}

// Built-in log sink: stderr with ANSI colors. Level filtering is the
// embedder's responsibility; every record reaches the sink.
void stderr_log_sink(unirt_LogLevel level, const char* message) {
    static constexpr struct {
        const char* tag;
        const char* color;
    } kStyles[] = {
        {"[TRACE] ", "\033[90m"},
        {"[DEBUG] ", "\033[34m"},
        {"[ INFO] ", "\033[32m"},
        {"[ WARN] ", "\033[33m"},
        {"[ERROR] ", "\033[31m"},
    };
    if (level < 0 || level > UNIRT_LOG_LEVEL_ERROR) return;
    const auto& style = kStyles[level];
    // One record at a time. The stream itself is safe to share -- the standard
    // says concurrent writes to it are not a data race -- but a record is four
    // separate insertions, and with several sequences decoding at once they
    // arrive shuffled into each other's colour codes. A log that has to be
    // reassembled by eye is worse than a lock nobody can measure.
    static std::mutex   sink_mutex;
    std::lock_guard<std::mutex> lock(sink_mutex);
    std::cerr << style.color << style.tag << message << "\033[0m" << std::endl;
}

}  // namespace

// The logging macros in logging.h resolve to these two symbols.
std::atomic<unirt_log_callback> unirt_log{stderr_log_sink};
unirt_LogLevel                  unirt_log_level = UNIRT_LOG_LEVEL_TRACE;

int32_t unirt_init(void) {
    auto&                       state = lifecycle();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.running) return UNIRT_ERROR_COMMON_ALREADY_INITIALIZED;

#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    UNIRT_LOG_DEBUG("starting unirt runtime");

    if (!host_cpu_is_capable()) {
        UNIRT_LOG_ERROR(
            "this CPU lacks features this unirt build requires; running would SIGILL");
        return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }

    return bridge::shielded("runtime.init", UNIRT_ERROR_COMMON_UNKNOWN, [&]() -> int32_t {
        try {
            PluginDirectory::instance().discover();
        } catch (...) {
            PluginDirectory::instance().unload_all();
            throw;
        }
        state.running = true;
        return UNIRT_SUCCESS;
    });
}

int32_t unirt_deinit(void) {
    auto&                       state = lifecycle();
    std::lock_guard<std::mutex> lock(state.mutex);
    UNIRT_LOG_DEBUG("stopping unirt runtime");

    if (!state.running) return UNIRT_ERROR_COMMON_NOT_INITIALIZED;

    // Unloading plugin libraries while models still reference their vtables
    // is a use-after-free; make callers destroy handles first.
    const std::size_t live = active_handle_count();
    if (live != 0) {
        UNIRT_LOG_ERROR("cannot deinit while {} model handle(s) remain open", live);
        return UNIRT_ERROR_COMMON_BUSY;
    }

    return bridge::shielded("runtime.deinit", UNIRT_ERROR_COMMON_UNKNOWN, [&]() -> int32_t {
        PluginDirectory::instance().unload_all();
        state.running = false;
        return UNIRT_SUCCESS;
    });
}

int32_t unirt_register_plugin(unirt_plugin_id_func identity, unirt_plugin_open_func open_plugin) {
    UNIRT_LOG_DEBUG("registering statically linked plugin");
    if (!identity || !open_plugin) {
        return bridge::fail(
            UNIRT_ERROR_COMMON_INVALID_INPUT, "register_plugin: identity/open function is NULL");
    }

    return bridge::shielded("runtime.register_plugin", UNIRT_ERROR_COMMON_PLUGIN_INVALID, [&]() -> int32_t {
        PluginDirectory::instance().adopt(identity, open_plugin);
        return UNIRT_SUCCESS;
    });
}

int32_t unirt_set_log(unirt_log_callback callback) {
    unirt_log.store(callback ? callback : stderr_log_sink, std::memory_order_release);
    return UNIRT_SUCCESS;
}

void unirt_free(void* ptr) {
    if (ptr) free(ptr);
}

const char* unirt_version() { return build_config::kBridgeVersion; }

const char* unirt_get_plugin_version(unirt_PluginId plugin_id) {
    if (!plugin_id) {
        UNIRT_LOG_ERROR("plugin_id is null");
        return nullptr;
    }
    try {
        BackendPackage* plugin = PluginDirectory::instance().get<BackendPackage>(plugin_id);
        return plugin ? plugin->version() : nullptr;
    } catch (const std::exception& e) {
        UNIRT_LOG_ERROR("cannot read version of '{}': {}", plugin_id, e.what());
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

int32_t unirt_get_plugin_list(unirt_GetPluginListOutput* output) {
    UNIRT_LOG_TRACE("listing plugins");
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *output = {};

    return bridge::shielded("runtime.plugin_list", UNIRT_ERROR_COMMON_UNKNOWN, [&]() -> int32_t {
        const auto ids = PluginDirectory::instance().ids();
        if (ids.empty()) return UNIRT_SUCCESS;
        if (ids.size() > static_cast<size_t>(INT32_MAX)) return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;

        // One allocation: a pointer table followed by the packed strings, so
        // unirt_free() on plugin_ids releases everything.
        size_t text_bytes = 0;
        for (const auto& id : ids) {
            if (text_bytes > SIZE_MAX - id.size() - 1) return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
            text_bytes += id.size() + 1;
        }
        const size_t table_bytes = ids.size() * sizeof(unirt_PluginId);
        if (text_bytes > SIZE_MAX - table_bytes) return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;

        auto* block = static_cast<unirt_PluginId*>(std::malloc(table_bytes + text_bytes));
        if (!block) return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;

        char* text_cursor = reinterpret_cast<char*>(block) + table_bytes;
        for (size_t i = 0; i < ids.size(); ++i) {
            const size_t span = ids[i].size() + 1;
            std::memcpy(text_cursor, ids[i].c_str(), span);
            block[i] = text_cursor;
            text_cursor += span;
        }
        output->plugin_ids   = block;
        output->plugin_count = static_cast<int32_t>(ids.size());
        return UNIRT_SUCCESS;
    });
}

int32_t unirt_get_plugin_modalities(unirt_PluginId plugin_id, uint32_t* out_modalities) {
    if (!out_modalities) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *out_modalities = 0;
    if (!plugin_id) {
        return bridge::fail(UNIRT_ERROR_COMMON_INVALID_INPUT, "plugin_modalities: plugin_id is NULL");
    }

    return bridge::shielded("runtime.plugin_modalities", UNIRT_ERROR_COMMON_UNKNOWN, [&]() -> int32_t {
        BackendPackage* plugin = PluginDirectory::instance().get<BackendPackage>(plugin_id);
        if (!plugin) return UNIRT_ERROR_COMMON_UNKNOWN;
        *out_modalities = plugin->modalities();
        return UNIRT_SUCCESS;
    });
}

int32_t unirt_get_device_list(const unirt_GetDeviceListInput* input, unirt_GetDeviceListOutput* output) {
    UNIRT_LOG_TRACE("listing devices");
    if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *output = {};
    if (!input || !input->plugin_id) return UNIRT_ERROR_COMMON_INVALID_INPUT;

    return bridge::shielded("runtime.device_list", UNIRT_ERROR_COMMON_UNKNOWN, [&]() -> int32_t {
        BackendPackage* plugin = PluginDirectory::instance().get<BackendPackage>(input->plugin_id);
        if (!plugin) return UNIRT_ERROR_COMMON_UNKNOWN;
        return plugin->get_device_list(input, output);
    });
}

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "backend_modules.h"

#ifdef UNIRT_GGML_BACKEND_DL

#include <filesystem>
#include <mutex>
#include <string>

#include <ggml-backend.h>

#include "logging.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace unirt::llama_plugin {

namespace {

// Anchors the self-location lookup below. It has to be a symbol belonging to
// *this* module: unirt::module_directory() would answer with libunirt's
// directory, which is one level up from where the backend modules install.
void anchor() {}

std::filesystem::path plugin_directory() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&anchor), &module)) {
        return {};
    }
    wchar_t path[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(module, path, MAX_PATH);
    if (length == 0 || length == MAX_PATH) return {};
    return std::filesystem::path(path).parent_path();
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&anchor), &info) || !info.dli_fname) return {};
    std::error_code ec;
    const auto resolved = std::filesystem::canonical(info.dli_fname, ec);
    if (ec) return {};
    return resolved.parent_path();
#endif
}

// The instruction-set features the loaded backend was compiled for, as
// "AVX2=1 FMA=1 ...". With one CPU backend per level shipped and ggml picking
// at load time, this is the only way a user can tell which one they got --
// and the first thing to ask when inference is slower than the same machine
// manages elsewhere. Empty when the backend does not report features.
std::string backend_features(ggml_backend_reg_t reg) {
    auto get_features = reinterpret_cast<ggml_backend_get_features_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features"));
    if (!get_features) return {};
    std::string listed;
    for (auto* feature = get_features(reg); feature && feature->name; ++feature) {
        if (!listed.empty()) listed += ' ';
        listed += feature->name;
        if (feature->value && *feature->value) {
            listed += '=';
            listed += feature->value;
        }
    }
    return listed;
}

}  // namespace

void load_ggml_backend_modules() {
    static std::once_flag once;
    std::call_once(once, [] {
        const auto directory = plugin_directory();
        if (directory.empty()) {
            UNIRT_LOG_ERROR(
                "llama_cpp: cannot locate this plugin on disk, so ggml's backend modules "
                "cannot be found; inference will have no backend at all");
            return;
        }
        // ggml parses this back with fs::u8path, so hand it UTF-8 rather than
        // the active code page's idea of the same path.
        const auto utf8 = directory.u8string();
        ggml_backend_load_all_from_path(reinterpret_cast<const char*>(utf8.c_str()));
        // Name them rather than count them: a module that ships but cannot load
        // -- Vulkan on a machine with no driver -- is skipped in silence by
        // design, and this line is the only thing that tells a user with a GPU
        // why their device list came back with just the CPU on it.
        std::string names;
        for (size_t index = 0; index < ggml_backend_reg_count(); ++index) {
            ggml_backend_reg_t reg  = ggml_backend_reg_get(index);
            const char*        name = ggml_backend_reg_name(reg);
            if (!names.empty()) names += ", ";
            names += name ? name : "?";
            const std::string features = backend_features(reg);
            if (!features.empty()) names += " (" + features + ")";
        }
        UNIRT_LOG_DEBUG(
            "llama_cpp: ggml backends registered from {}: {}",
            reinterpret_cast<const char*>(utf8.c_str()), names.empty() ? "none" : names);
    });
}

}  // namespace unirt::llama_plugin

#else

namespace unirt::llama_plugin {

// Backends are linked into libggml and register themselves; nothing to load.
void load_ggml_backend_modules() {}

}  // namespace unirt::llama_plugin

#endif  // UNIRT_GGML_BACKEND_DL

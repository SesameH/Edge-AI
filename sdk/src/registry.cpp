// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "registry.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <windows.h>

#include <vector>
#else
#include <dlfcn.h>
#endif

#include "logging.h"
#include "plugin_adapters.h"
#include "utils.h"

namespace unirt {

namespace {

const char* platform_plugin_filename() {
#if defined(_WIN32)
    return "unirt_plugin.dll";
#elif defined(__APPLE__)
    return "libunirt_plugin.dylib";
#else
    return "libunirt_plugin.so";
#endif
}

// Flat layouts (Android app lib dirs, single-directory installs) name each
// plugin `<prefix><id><suffix>` beside libunirt instead of nesting it in a
// per-plugin subdirectory.
const char* flat_plugin_prefix() {
#if defined(_WIN32)
    return "unirt_plugin_";
#else
    return "libunirt_plugin_";
#endif
}

const char* flat_plugin_suffix() {
#if defined(_WIN32)
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
// DynamicLibrary

DynamicLibrary::DynamicLibrary(const std::filesystem::path& path) {
#if defined(_WIN32)
    const auto  absolute = std::filesystem::absolute(path);
    const DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_APPLICATION_DIR |
                        LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_SYSTEM32;
    native_ = LoadLibraryExW(absolute.wstring().c_str(), nullptr, flags);
    if (!native_) {
        throw std::runtime_error("LoadLibraryExW failed: " + std::to_string(GetLastError()));
    }
#else
    native_ = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!native_) {
        throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    }
#endif
}

void DynamicLibrary::close() noexcept {
    if (!native_) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(native_));
#else
    dlclose(native_);
#endif
    native_ = nullptr;
}

DynamicLibrary::~DynamicLibrary() { close(); }

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : native_(std::exchange(other.native_, nullptr)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        close();
        native_ = std::exchange(other.native_, nullptr);
    }
    return *this;
}

void* DynamicLibrary::symbol(const char* name) const noexcept {
    if (!native_) return nullptr;
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(native_), name));
#else
    return dlsym(native_, name);
#endif
}

// ---------------------------------------------------------------------------
// PluginSlot

PluginSlot PluginSlot::from_library(const std::filesystem::path& path) {
    UNIRT_LOG_TRACE("loading plugin candidate {}", path.u8string());
    PluginSlot slot;
    slot.library_ = DynamicLibrary(path);

    // ABI stamp first: never call into a plugin whose vtable layout may
    // differ from ours.
    auto* abi_symbol = slot.library_.symbol("unirt_plugin_abi_version");
    if (!abi_symbol) throw std::runtime_error("plugin lacks unirt_plugin_abi_version()");
    const uint32_t abi = reinterpret_cast<uint32_t (*)()>(abi_symbol)();
    if (abi != UNIRT_PLUGIN_ABI_VERSION) {
        throw std::runtime_error(
            "plugin ABI " + std::to_string(abi) + " does not match runtime ABI " +
            std::to_string(UNIRT_PLUGIN_ABI_VERSION));
    }

    auto* identity_symbol = slot.library_.symbol("unirt_plugin_id");
    if (!identity_symbol) throw std::runtime_error("plugin lacks unirt_plugin_id()");
    const char* id = reinterpret_cast<IdentityFn>(identity_symbol)();
    if (!id || !id[0]) throw std::runtime_error("unirt_plugin_id() returned an empty id");
    slot.id_ = id;

    auto* factory_symbol = slot.library_.symbol("unirt_plugin_open");
    if (!factory_symbol) throw std::runtime_error("plugin lacks unirt_plugin_open()");
    slot.factory_ = reinterpret_cast<FactoryFn>(factory_symbol);

    UNIRT_LOG_TRACE("plugin '{}' passed load checks", slot.id_);
    return slot;
}

PluginSlot PluginSlot::from_functions(IdentityFn identity, FactoryFn factory) {
    if (!identity || !factory) throw std::runtime_error("null plugin registration functions");
    const char* id = identity();
    if (!id || !id[0]) throw std::runtime_error("unirt_plugin_id() returned an empty id");

    PluginSlot slot;
    slot.id_      = id;
    slot.factory_ = factory;
    return slot;
}

BackendPackage& PluginSlot::instance() {
    if (!plugin_) {
        unirt_PluginTable* table = factory_ ? factory_() : nullptr;
        if (!table) throw std::runtime_error("unirt_plugin_open() returned null");
        if (!bridge::table_complete(table)) {
            if (table->destroy) table->destroy(table->self);
            throw std::runtime_error("plugin table is smaller than the runtime's contract");
        }
        plugin_ = std::make_unique<bridge::TablePackage>(table);
    }
    return *plugin_;
}

// ---------------------------------------------------------------------------
// PluginDirectory

PluginDirectory& PluginDirectory::instance() {
    static PluginDirectory directory;
    return directory;
}

std::filesystem::path PluginDirectory::plugin_root() const {
#if defined(_WIN32)
    size_t required = 0;
    _wgetenv_s(&required, nullptr, 0, L"UNIRT_PLUGIN_PATH");
    if (required > 0) {
        std::vector<wchar_t> value(required);
        _wgetenv_s(&required, value.data(), required, L"UNIRT_PLUGIN_PATH");
        if (value[0] != L'\0') return std::filesystem::path(value.data());
    }
#else
    if (const char* override_path = std::getenv("UNIRT_PLUGIN_PATH"); override_path && override_path[0]) {
        UNIRT_LOG_TRACE("plugin root from UNIRT_PLUGIN_PATH: {}", override_path);
        return std::filesystem::path(override_path);
    }
#endif
    const auto fallback = module_directory();
    UNIRT_LOG_TRACE("plugin root from shared library location: {}", fallback.u8string());
    // Publish the resolved root so subprocesses and diagnostics see the
    // effective value.
#if defined(_WIN32)
    _wputenv_s(L"UNIRT_PLUGIN_PATH", fallback.wstring().c_str());
#else
    setenv("UNIRT_PLUGIN_PATH", fallback.c_str(), 1);
#endif
    return fallback;
}

void PluginDirectory::discover() {
    const auto root = plugin_root();
    UNIRT_LOG_TRACE("scanning {} for plugins", root.u8string());

    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        std::filesystem::path candidate;
        if (entry.is_directory()) {
            candidate = entry.path() / platform_plugin_filename();
            if (!std::filesystem::is_regular_file(candidate)) continue;
        } else if (entry.is_regular_file()) {
            const std::string name = entry.path().filename().string();
            if (name.rfind(flat_plugin_prefix(), 0) != 0 ||
                entry.path().extension() != flat_plugin_suffix()) {
                continue;
            }
            candidate = entry.path();
        } else {
            continue;
        }

        try {
            PluginSlot slot = PluginSlot::from_library(candidate);
            std::string id  = slot.id();
            slots_.insert_or_assign(std::move(id), std::move(slot));
        } catch (const std::exception& e) {
            broken_.push_back(entry.path().filename().u8string());
            UNIRT_LOG_ERROR("plugin {} rejected: {}", candidate.u8string(), e.what());
        }
    }
    UNIRT_LOG_TRACE("plugin scan complete: {} loaded, {} rejected", slots_.size(), broken_.size());
}

void PluginDirectory::adopt(PluginSlot::IdentityFn identity, PluginSlot::FactoryFn factory) {
    PluginSlot slot = PluginSlot::from_functions(identity, factory);

    std::lock_guard<std::mutex> lock(mutex_);
    std::string                 id = slot.id();
    UNIRT_LOG_TRACE("adopting statically linked plugin '{}'", id);
    slots_.insert_or_assign(std::move(id), std::move(slot));
}

void PluginDirectory::unload_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    UNIRT_LOG_TRACE("unloading {} plugins", slots_.size());
    slots_.clear();
    broken_.clear();
}

std::vector<std::string> PluginDirectory::ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string>    listed;
    listed.reserve(slots_.size());
    for (const auto& [id, slot] : slots_) listed.push_back(id);
    return listed;
}

}  // namespace unirt

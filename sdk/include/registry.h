// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "plugin/backend_package.h"

namespace unirt {

/** Requested plugin id is not registered. */
class PluginNotFoundException : public std::exception {
   public:
    const char* what() const noexcept override { return "plugin not registered"; }
};

/** BackendPackage was found on disk but could not be loaded (dlopen / symbol / ABI). */
class PluginLoadException : public std::exception {
   public:
    const char* what() const noexcept override { return "plugin failed to load"; }
};

/** RAII wrapper over one dlopen'd (or LoadLibrary'd) shared object. */
class DynamicLibrary {
   public:
    DynamicLibrary() = default;
    explicit DynamicLibrary(const std::filesystem::path& path);
    ~DynamicLibrary();
    DynamicLibrary(DynamicLibrary&& other) noexcept;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;
    DynamicLibrary(const DynamicLibrary&)            = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    /** Resolve a symbol; returns nullptr when missing. */
    void* symbol(const char* name) const noexcept;
    bool  is_open() const noexcept { return native_ != nullptr; }

   private:
    void close() noexcept;
    void* native_ = nullptr;
};

/** One registered backend: its library (when dynamically loaded), its
 *  factory, and the lazily opened package (a C table from plugin_abi.h,
 *  wrapped back into BackendPackage by the bridge adapters). */
class PluginSlot {
   public:
    using FactoryFn  = unirt_PluginTable* (*)();
    using IdentityFn = const char* (*)();

    /** Load from a shared object; verifies the ABI stamp and reads the id.
     *  Throws std::runtime_error with a reason on any failure. */
    static PluginSlot from_library(const std::filesystem::path& path);

    const std::string& id() const noexcept { return id_; }

    /** The package, opened on first use: calls the factory, validates the
     *  table's struct_size, wraps it in a bridge adapter. Throws on a null
     *  or short table. */
    BackendPackage& instance();

    PluginSlot(PluginSlot&&) noexcept            = default;
    PluginSlot& operator=(PluginSlot&&) noexcept = default;

   private:
    PluginSlot() = default;

    DynamicLibrary          library_;
    FactoryFn               factory_ = nullptr;
    std::string             id_;
    std::unique_ptr<BackendPackage> plugin_;  // TablePackage adapter over the C table
};

/**
 * Process-wide catalogue of backends. Thread-safe. Plugins that fail to
 * load are remembered so lookups can distinguish "never existed" from
 * "exists but is broken".
 */
class PluginDirectory {
   public:
    static PluginDirectory& instance();

    /** Scan the plugin root (UNIRT_PLUGIN_PATH or the directory holding
     *  libunirt) for `<subdir>/<platform plugin name>` and load each. */
    void discover();

    /** Destroy every plugin instance and unload their libraries. */
    void unload_all();

    std::vector<std::string> ids() const;

    /** Fetch a plugin (or one of its modality interfaces) by id.
     *  Throws PluginNotFoundException / PluginLoadException. The returned
     *  modality pointers are owned by the caller; the BackendPackage pointer is
     *  owned by the directory. */
    template <typename Wanted>
    Wanted* get(const std::string& plugin_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        found = slots_.find(plugin_id);
        if (found == slots_.end()) {
            for (const auto& broken_id : broken_) {
                if (broken_id == plugin_id) throw PluginLoadException();
            }
            throw PluginNotFoundException();
        }
        BackendPackage& plugin = found->second.instance();
        if constexpr (std::is_same_v<Wanted, BackendPackage>) {
            return &plugin;
        } else if constexpr (std::is_same_v<Wanted, LlmBackend>) {
            return plugin.create_llm();
        } else if constexpr (std::is_same_v<Wanted, VlmBackend>) {
            return plugin.create_vlm();
        } else if constexpr (std::is_same_v<Wanted, EmbeddingBackend>) {
            return plugin.create_embedding();
        } else {
            static_assert(
                std::is_same_v<Wanted, BackendPackage>, "unsupported interface requested from PluginDirectory");
        }
    }

   private:
    PluginDirectory() = default;

    std::filesystem::path plugin_root() const;

    mutable std::mutex          mutex_;
    std::map<std::string, PluginSlot> slots_;
    std::vector<std::string>          broken_;
};

}  // namespace unirt

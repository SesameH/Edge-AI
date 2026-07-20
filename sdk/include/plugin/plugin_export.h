// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

/**
 * Plugin-side glue: turn a C++ BackendPackage implementation into the C
 * tables that actually cross the boundary (plugin_abi.h).
 *
 * Every thunk catches everything — exceptions must not unwind across the
 * module boundary — and destroy() runs in this module, so allocation and
 * release never straddle two heaps.
 *
 * A plugin's whole export surface becomes:
 *
 *   unirt_PluginId unirt_plugin_id() { return "my_backend"; }
 *   uint32_t unirt_plugin_abi_version() { return UNIRT_PLUGIN_ABI_VERSION; }
 *   unirt_PluginTable* unirt_plugin_open() {
 *       return unirt::plugin_export::open_package<MyPackage>();
 *   }
 */

#include <memory>
#include <new>

#include "backend_package.h"

namespace unirt::plugin_export {

/** Translate the in-flight exception into an error code. */
inline int32_t current_exception_code() noexcept {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
    } catch (...) {
        return UNIRT_ERROR_COMMON_UNKNOWN;
    }
}

namespace detail {

// Table and implementation live in one allocation, so `self` is enough to
// release both.
template <typename Table, typename Impl>
struct Holder {
    Table                 table{};
    std::unique_ptr<Impl> impl;
};

// One thunk shape fits every table entry: unwrap self, forward, catch.
#define UNIRT_PLUGIN_THUNK(holder_t, method)                                            \
    [](void* self, auto*... args) noexcept -> int32_t {                                 \
        try {                                                                           \
            return static_cast<holder_t*>(self)->impl->method(args...);                 \
        } catch (...) {                                                                 \
            return ::unirt::plugin_export::current_exception_code();                    \
        }                                                                               \
    }

}  // namespace detail

inline unirt_LlmTable* wrap(std::unique_ptr<LlmBackend> impl) noexcept {
    using Holder = detail::Holder<unirt_LlmTable, LlmBackend>;
    if (!impl) return nullptr;
    auto* holder = new (std::nothrow) Holder;
    if (!holder) return nullptr;
    holder->impl = std::move(impl);

    unirt_LlmTable& t        = holder->table;
    t.struct_size            = sizeof(unirt_LlmTable);
    t.self                   = holder;
    t.create                 = UNIRT_PLUGIN_THUNK(Holder, create);
    t.reset                  = [](void* self) noexcept -> int32_t {
        try {
            return static_cast<Holder*>(self)->impl->reset();
        } catch (...) {
            return current_exception_code();
        }
    };
    t.save_kv_cache          = UNIRT_PLUGIN_THUNK(Holder, save_kv_cache);
    t.load_kv_cache          = UNIRT_PLUGIN_THUNK(Holder, load_kv_cache);
    t.apply_chat_template    = UNIRT_PLUGIN_THUNK(Holder, apply_chat_template);
    t.generate               = UNIRT_PLUGIN_THUNK(Holder, generate);
    t.get_model_info         = UNIRT_PLUGIN_THUNK(Holder, get_model_info);
    t.get_runtime_stats      = UNIRT_PLUGIN_THUNK(Holder, get_runtime_stats);
    t.destroy                = [](void* self) noexcept { delete static_cast<Holder*>(self); };
    return &t;
}

inline unirt_VlmTable* wrap(std::unique_ptr<VlmBackend> impl) noexcept {
    using Holder = detail::Holder<unirt_VlmTable, VlmBackend>;
    if (!impl) return nullptr;
    auto* holder = new (std::nothrow) Holder;
    if (!holder) return nullptr;
    holder->impl = std::move(impl);

    unirt_VlmTable& t     = holder->table;
    t.struct_size         = sizeof(unirt_VlmTable);
    t.self                = holder;
    t.create              = UNIRT_PLUGIN_THUNK(Holder, create);
    t.reset               = [](void* self) noexcept -> int32_t {
        try {
            return static_cast<Holder*>(self)->impl->reset();
        } catch (...) {
            return current_exception_code();
        }
    };
    t.apply_chat_template = UNIRT_PLUGIN_THUNK(Holder, apply_chat_template);
    t.generate            = UNIRT_PLUGIN_THUNK(Holder, generate);
    t.get_capabilities    = UNIRT_PLUGIN_THUNK(Holder, get_capabilities);
    t.destroy             = [](void* self) noexcept { delete static_cast<Holder*>(self); };
    return &t;
}

inline unirt_EmbeddingTable* wrap(std::unique_ptr<EmbeddingBackend> impl) noexcept {
    using Holder = detail::Holder<unirt_EmbeddingTable, EmbeddingBackend>;
    if (!impl) return nullptr;
    auto* holder = new (std::nothrow) Holder;
    if (!holder) return nullptr;
    holder->impl = std::move(impl);

    unirt_EmbeddingTable& t = holder->table;
    t.struct_size           = sizeof(unirt_EmbeddingTable);
    t.self                  = holder;
    t.create                = UNIRT_PLUGIN_THUNK(Holder, create);
    t.encode                = UNIRT_PLUGIN_THUNK(Holder, encode);
    t.get_runtime_stats     = UNIRT_PLUGIN_THUNK(Holder, get_runtime_stats);
    t.destroy               = [](void* self) noexcept { delete static_cast<Holder*>(self); };
    return &t;
}

inline unirt_PluginTable* wrap(std::unique_ptr<BackendPackage> impl) noexcept {
    using Holder = detail::Holder<unirt_PluginTable, BackendPackage>;
    if (!impl) return nullptr;
    auto* holder = new (std::nothrow) Holder;
    if (!holder) return nullptr;
    holder->impl = std::move(impl);

    unirt_PluginTable& t = holder->table;
    t.struct_size        = sizeof(unirt_PluginTable);
    t.self               = holder;
    t.version            = [](void* self) noexcept -> const char* {
        try {
            return static_cast<Holder*>(self)->impl->version();
        } catch (...) {
            return "unknown";
        }
    };
    t.get_device_list  = UNIRT_PLUGIN_THUNK(Holder, get_device_list);
    t.create_llm       = [](void* self) noexcept -> unirt_LlmTable* {
        try {
            return wrap(std::unique_ptr<LlmBackend>(static_cast<Holder*>(self)->impl->create_llm()));
        } catch (...) {
            return nullptr;
        }
    };
    t.create_vlm       = [](void* self) noexcept -> unirt_VlmTable* {
        try {
            return wrap(std::unique_ptr<VlmBackend>(static_cast<Holder*>(self)->impl->create_vlm()));
        } catch (...) {
            return nullptr;
        }
    };
    t.create_embedding = [](void* self) noexcept -> unirt_EmbeddingTable* {
        try {
            return wrap(std::unique_ptr<EmbeddingBackend>(
                static_cast<Holder*>(self)->impl->create_embedding()));
        } catch (...) {
            return nullptr;
        }
    };
    t.destroy          = [](void* self) noexcept { delete static_cast<Holder*>(self); };
    try {
        t.modalities = holder->impl->modalities();
    } catch (...) {
        t.modalities = 0;
    }
    return &t;
}

#undef UNIRT_PLUGIN_THUNK

/** One-liner for unirt_plugin_open(): construct the package and wrap it. */
template <typename Package, typename... Args>
unirt_PluginTable* open_package(Args&&... args) noexcept {
    try {
        return wrap(std::unique_ptr<BackendPackage>(
            std::make_unique<Package>(std::forward<Args>(args)...)));
    } catch (...) {
        return nullptr;
    }
}

}  // namespace unirt::plugin_export

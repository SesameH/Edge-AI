// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

/**
 * The plugin boundary contract — pure C.
 *
 * A plugin is a shared object exporting exactly three C symbols:
 *
 *   unirt_PluginId     unirt_plugin_id(void);          // cheap, no state
 *   uint32_t           unirt_plugin_abi_version(void); // == UNIRT_PLUGIN_ABI_VERSION
 *   unirt_PluginTable* unirt_plugin_open(void);        // the package itself
 *
 * Everything that crosses the runtime/plugin boundary afterwards is a C
 * struct of function pointers defined in this header. No C++ types, no
 * vtables, no exceptions cross the boundary, so runtime and plugins may be
 * built with different compilers, stdlibs, or languages.
 *
 * Rules both sides rely on:
 *  - `struct_size` is stamped by the producer with sizeof() of the struct it
 *    was compiled against. A consumer must not touch fields beyond the
 *    producer's struct_size. New fields go at the end; adding one therefore
 *    does NOT require an ABI bump — only reshaping existing fields does.
 *  - `self` is the producer's private state; callers pass it back verbatim.
 *  - `destroy(self)` releases self AND the table itself, inside the module
 *    that allocated them. It must be the last call on a table.
 *  - Optional operations may be NULL; required ones (create/generate/encode/
 *    destroy) must be present.
 *  - Functions must not let exceptions escape; failures are int32_t
 *    unirt error codes.
 */

#include <stddef.h>
#include <stdint.h>

#include "unirt.h"

// Both sides of the boundary must agree on the shape of the tables below.
// Bump when reshaping existing fields; the loader refuses plugins whose
// stamp differs before any table is opened.
#define UNIRT_PLUGIN_ABI_VERSION 4u

#if defined(_WIN32)
#define UNIRT_PLUGIN_API __declspec(dllexport)
#else
#define UNIRT_PLUGIN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Text generation. One table instance = one loaded model + decode state. */
typedef struct unirt_LlmTable {
    size_t struct_size;
    void*  self;
    int32_t (*create)(void* self, const unirt_LlmCreateInput* input);
    int32_t (*reset)(void* self);
    int32_t (*save_kv_cache)(void* self, const unirt_KvCacheSaveInput* input, unirt_KvCacheSaveOutput* output);
    int32_t (*load_kv_cache)(void* self, const unirt_KvCacheLoadInput* input, unirt_KvCacheLoadOutput* output);
    int32_t (*apply_chat_template)(
        void* self, const unirt_LlmApplyChatTemplateInput* input, unirt_LlmApplyChatTemplateOutput* output);
    int32_t (*generate)(void* self, const unirt_LlmGenerateInput* input, unirt_LlmGenerateOutput* output);
    int32_t (*get_model_info)(void* self, unirt_LlmModelInfo* output);
    int32_t (*get_runtime_stats)(void* self, unirt_LlmRuntimeStats* output);
    void (*destroy)(void* self);
} unirt_LlmTable;

/** Multimodal generation. */
typedef struct unirt_VlmTable {
    size_t struct_size;
    void*  self;
    int32_t (*create)(void* self, const unirt_VlmCreateInput* input);
    int32_t (*reset)(void* self);
    int32_t (*apply_chat_template)(
        void* self, const unirt_VlmApplyChatTemplateInput* input, unirt_VlmApplyChatTemplateOutput* output);
    int32_t (*generate)(void* self, const unirt_VlmGenerateInput* input, unirt_VlmGenerateOutput* output);
    int32_t (*get_capabilities)(void* self, unirt_VlmCapabilities* output);
    void (*destroy)(void* self);
    /* -- Appended fields (guarded by struct_size; never reorder above) -- */
    int32_t (*get_runtime_stats)(void* self, unirt_VlmRuntimeStats* output);
} unirt_VlmTable;

/** Embedding encoder. */
typedef struct unirt_EmbeddingTable {
    size_t struct_size;
    void*  self;
    int32_t (*create)(void* self, const unirt_EmbeddingCreateInput* input);
    int32_t (*encode)(void* self, const unirt_EmbeddingEncodeInput* input, unirt_EmbeddingEncodeOutput* output);
    int32_t (*get_runtime_stats)(void* self, unirt_EmbeddingRuntimeStats* output);
    void (*destroy)(void* self);
    /* -- Appended fields (guarded by struct_size; never reorder above) -- */
    int32_t (*rerank)(void* self, const unirt_EmbeddingRerankInput* input, unirt_EmbeddingRerankOutput* output);
} unirt_EmbeddingTable;

/** The package: identity/devices plus factories for the modalities the
 *  plugin supports. Factories return NULL when unsupported or on failure. */
typedef struct unirt_PluginTable {
    size_t struct_size;
    void*  self;
    const char* (*version)(void* self);
    int32_t (*get_device_list)(
        void* self, const unirt_GetDeviceListInput* input, unirt_GetDeviceListOutput* output);
    unirt_LlmTable* (*create_llm)(void* self);
    unirt_VlmTable* (*create_vlm)(void* self);
    unirt_EmbeddingTable* (*create_embedding)(void* self);
    void (*destroy)(void* self);
    /* -- Appended fields (guarded by struct_size; never reorder above) -- */
    uint32_t modalities; /**< UNIRT_MODALITY_* bits; 0 = undeclared */
} unirt_PluginTable;

UNIRT_PLUGIN_API unirt_PluginId unirt_plugin_id(void);
UNIRT_PLUGIN_API uint32_t unirt_plugin_abi_version(void);
UNIRT_PLUGIN_API unirt_PluginTable* unirt_plugin_open(void);

#ifdef __cplusplus
}
#endif

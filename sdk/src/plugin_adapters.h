// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

/**
 * Bridge-side glue: wrap the C tables a plugin exports (plugin_abi.h) back
 * into the C++ modality interfaces the bridge programs against. Mirror of
 * plugin_export.h on the other side of the boundary.
 *
 * Each adapter owns its table: the destructor calls table->destroy(self),
 * which releases the plugin-side state inside the plugin's own module.
 * NULL optional entries surface as PARAM_NOT_SUPPORTED, matching the C++
 * base-class defaults. Tables whose struct_size is smaller than the
 * contract this bridge was compiled against are refused at wrap time.
 */

#include <cstddef>
#include <type_traits>

#include "plugin/backend_package.h"

namespace unirt::bridge {

/** The bytes a producer must have stamped for us to trust its table. For
 *  tables with appended optional fields this is the offset of the first
 *  appended field, NOT sizeof — older producers that predate the appended
 *  fields must keep loading. */
template <typename Table>
inline constexpr std::size_t kRequiredTableBytes = sizeof(Table);

template <>
inline constexpr std::size_t kRequiredTableBytes<unirt_PluginTable> =
    offsetof(unirt_PluginTable, modalities);

template <>
inline constexpr std::size_t kRequiredTableBytes<unirt_VlmTable> =
    offsetof(unirt_VlmTable, get_runtime_stats);

template <typename Table>
bool table_complete(const Table* table) noexcept {
    return table && table->struct_size >= kRequiredTableBytes<Table>;
}

/** True when the producer's struct_size covers a given appended field. */
#define UNIRT_TABLE_HAS_FIELD(table_ptr, field) \
    ((table_ptr)->struct_size >= offsetof(std::remove_pointer_t<decltype(table_ptr)>, field) + \
                                     sizeof((table_ptr)->field))

class TableLlm final : public LlmBackend {
   public:
    explicit TableLlm(unirt_LlmTable* table) : t_(table) {}
    ~TableLlm() override {
        if (t_ && t_->destroy) t_->destroy(t_->self);
    }
    TableLlm(const TableLlm&)            = delete;
    TableLlm& operator=(const TableLlm&) = delete;

    int32_t create(const unirt_LlmCreateInput* input) override {
        return t_->create ? t_->create(t_->self, input) : UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    int32_t reset() override {
        return t_->reset ? t_->reset(t_->self) : UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    int32_t save_kv_cache(const unirt_KvCacheSaveInput* input, unirt_KvCacheSaveOutput* output) override {
        return t_->save_kv_cache ? t_->save_kv_cache(t_->self, input, output)
                                 : UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    int32_t load_kv_cache(const unirt_KvCacheLoadInput* input, unirt_KvCacheLoadOutput* output) override {
        return t_->load_kv_cache ? t_->load_kv_cache(t_->self, input, output)
                                 : UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    int32_t apply_chat_template(
        const unirt_LlmApplyChatTemplateInput* input, unirt_LlmApplyChatTemplateOutput* output) override {
        return t_->apply_chat_template ? t_->apply_chat_template(t_->self, input, output)
                                       : UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    int32_t generate(const unirt_LlmGenerateInput* input, unirt_LlmGenerateOutput* output) override {
        return t_->generate ? t_->generate(t_->self, input, output) : UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    int32_t get_model_info(unirt_LlmModelInfo* output) override {
        return t_->get_model_info ? t_->get_model_info(t_->self, output)
                                  : UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    int32_t get_runtime_stats(unirt_LlmRuntimeStats* output) override {
        return t_->get_runtime_stats ? t_->get_runtime_stats(t_->self, output)
                                     : UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }

   private:
    unirt_LlmTable* t_;
};

class TableVlm final : public VlmBackend {
   public:
    explicit TableVlm(unirt_VlmTable* table) : t_(table) {}
    ~TableVlm() override {
        if (t_ && t_->destroy) t_->destroy(t_->self);
    }
    TableVlm(const TableVlm&)            = delete;
    TableVlm& operator=(const TableVlm&) = delete;

    int32_t create(const unirt_VlmCreateInput* input) override {
        return t_->create ? t_->create(t_->self, input) : UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    int32_t reset() override {
        return t_->reset ? t_->reset(t_->self) : UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    int32_t apply_chat_template(
        const unirt_VlmApplyChatTemplateInput* input, unirt_VlmApplyChatTemplateOutput* output) override {
        return t_->apply_chat_template ? t_->apply_chat_template(t_->self, input, output)
                                       : UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    int32_t generate(const unirt_VlmGenerateInput* input, unirt_VlmGenerateOutput* output) override {
        return t_->generate ? t_->generate(t_->self, input, output) : UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    int32_t get_capabilities(unirt_VlmCapabilities* output) override {
        if (t_->get_capabilities) return t_->get_capabilities(t_->self, output);
        return VlmBackend::get_capabilities(output);
    }
    int32_t get_runtime_stats(unirt_VlmRuntimeStats* output) override {
        if (UNIRT_TABLE_HAS_FIELD(t_, get_runtime_stats) && t_->get_runtime_stats) {
            return t_->get_runtime_stats(t_->self, output);
        }
        return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }

   private:
    unirt_VlmTable* t_;
};

class TableEmbedding final : public EmbeddingBackend {
   public:
    explicit TableEmbedding(unirt_EmbeddingTable* table) : t_(table) {}
    ~TableEmbedding() override {
        if (t_ && t_->destroy) t_->destroy(t_->self);
    }
    TableEmbedding(const TableEmbedding&)            = delete;
    TableEmbedding& operator=(const TableEmbedding&) = delete;

    int32_t create(const unirt_EmbeddingCreateInput* input) override {
        return t_->create ? t_->create(t_->self, input) : UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    int32_t encode(const unirt_EmbeddingEncodeInput* input, unirt_EmbeddingEncodeOutput* output) override {
        return t_->encode ? t_->encode(t_->self, input, output) : UNIRT_ERROR_COMMON_NOT_SUPPORTED;
    }
    int32_t get_runtime_stats(unirt_EmbeddingRuntimeStats* output) override {
        return t_->get_runtime_stats ? t_->get_runtime_stats(t_->self, output)
                                     : UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }

   private:
    unirt_EmbeddingTable* t_;
};

class TablePackage final : public BackendPackage {
   public:
    explicit TablePackage(unirt_PluginTable* table) : t_(table) {}
    ~TablePackage() override {
        if (t_ && t_->destroy) t_->destroy(t_->self);
    }
    TablePackage(const TablePackage&)            = delete;
    TablePackage& operator=(const TablePackage&) = delete;

    const char* version() override {
        const char* v = t_->version ? t_->version(t_->self) : nullptr;
        return v ? v : "unknown";
    }

    uint32_t modalities() override {
        return UNIRT_TABLE_HAS_FIELD(t_, modalities) ? t_->modalities : 0u;
    }

    int32_t get_device_list(
        const unirt_GetDeviceListInput* input, unirt_GetDeviceListOutput* output) override {
        if (t_->get_device_list) return t_->get_device_list(t_->self, input, output);
        return BackendPackage::get_device_list(input, output);
    }

    LlmBackend* create_llm() override {
        return adopt<TableLlm>(t_->create_llm ? t_->create_llm(t_->self) : nullptr);
    }
    VlmBackend* create_vlm() override {
        return adopt<TableVlm>(t_->create_vlm ? t_->create_vlm(t_->self) : nullptr);
    }
    EmbeddingBackend* create_embedding() override {
        return adopt<TableEmbedding>(t_->create_embedding ? t_->create_embedding(t_->self) : nullptr);
    }

   private:
    /** Wrap a freshly minted modality table, refusing short ones. */
    template <typename Adapter, typename Table>
    static Adapter* adopt(Table* table) {
        if (!table) return nullptr;
        if (!table_complete(table)) {
            if (table->destroy) table->destroy(table->self);
            return nullptr;
        }
        return new Adapter(table);
    }

    unirt_PluginTable* t_;
};

}  // namespace unirt::bridge

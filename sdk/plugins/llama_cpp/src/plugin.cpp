// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "plugin/plugin_export.h"

#include <cstdlib>
#include <limits>
#include <mutex>
#include <string_view>
#include <vector>

#include <llama.h>
#include <mtmd-helper.h>

#include "backend_modules.h"
#include "build_config.h"
#include "embedding.h"
#include "llm.h"
#include "vlm.h"
#include "logging.h"

namespace unirt::llama_plugin {

namespace {

std::mutex backend_mutex;
size_t     backend_users = 0;

void llama_log_bridge(ggml_log_level level, const char* message, void*) {
    std::string_view text = message ? message : "";
    while (!text.empty() && (text.back() == '\r' || text.back() == '\n')) {
        text.remove_suffix(1);
    }
    if (text.empty()) return;

    switch (level) {
        case GGML_LOG_LEVEL_DEBUG:
            UNIRT_LOG_TRACE("llama.cpp: {}", text);
            break;
        case GGML_LOG_LEVEL_INFO:
            UNIRT_LOG_DEBUG("llama.cpp: {}", text);
            break;
        case GGML_LOG_LEVEL_WARN:
            UNIRT_LOG_WARN("llama.cpp: {}", text);
            break;
        case GGML_LOG_LEVEL_ERROR:
            UNIRT_LOG_ERROR("llama.cpp: {}", text);
            break;
        default:
            UNIRT_LOG_TRACE("llama.cpp: {}", text);
            break;
    }
}

void acquire_backend() {
    std::lock_guard<std::mutex> lock(backend_mutex);
    if (backend_users++ == 0) {
        llama_log_set(llama_log_bridge, nullptr);
        mtmd_helper_log_set(llama_log_bridge, nullptr);
        // Before llama_backend_init(), which otherwise runs ggml's own default
        // module search (executable directory, working directory) and finds
        // nothing -- ours live beside this plugin. Registering first also makes
        // that fallback a no-op, since it only runs on an empty registry.
        load_ggml_backend_modules();
        llama_backend_init();
    }
}

void release_backend() {
    std::lock_guard<std::mutex> lock(backend_mutex);
    if (backend_users == 0 || --backend_users != 0) return;
    llama_backend_free();
    mtmd_helper_log_set(nullptr, nullptr);
    llama_log_set(nullptr, nullptr);
}

}  // namespace

class LlamaCppPlugin final : public BackendPackage {
   public:
    LlamaCppPlugin() { acquire_backend(); }
    ~LlamaCppPlugin() override { release_backend(); }

    const char* version() override { return "llama.cpp-public-api/1"; }

    uint32_t modalities() override {
        return UNIRT_MODALITY_LLM | UNIRT_MODALITY_VLM | UNIRT_MODALITY_EMBEDDING;
    }

    int32_t get_device_list(
        const unirt_GetDeviceListInput* input,
        unirt_GetDeviceListOutput* output) override {
        if (!input || !output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
        *output = {};

        const size_t backend_count = ggml_backend_dev_count();
        std::vector<ggml_backend_dev_t> devices;
        devices.reserve(backend_count);
        for (size_t index = 0; index < backend_count; ++index) {
            ggml_backend_dev_t device = ggml_backend_dev_get(index);
            const char* id = device ? ggml_backend_dev_name(device) : nullptr;
            const char* name = device ? ggml_backend_dev_description(device) : nullptr;
            if ((id && id[0]) || (name && name[0])) devices.push_back(device);
        }
        const size_t count = devices.size();
        if (count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return UNIRT_ERROR_COMMON_UNKNOWN;
        }
        if (count == 0) return UNIRT_SUCCESS;

        auto** ids = static_cast<const char**>(std::calloc(count, sizeof(const char*)));
        auto** names = static_cast<const char**>(std::calloc(count, sizeof(const char*)));
        if (!ids || !names) {
            std::free(ids);
            std::free(names);
            return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
        }

        for (size_t index = 0; index < count; ++index) {
            ggml_backend_dev_t device = devices[index];
            const char* id = ggml_backend_dev_name(device);
            const char* name = ggml_backend_dev_description(device);
            ids[index] = id ? id : "";
            names[index] = name ? name : ids[index];
        }
        output->device_ids = ids;
        output->device_names = names;
        output->device_count = static_cast<int32_t>(count);
        return UNIRT_SUCCESS;
    }

    LlmBackend* create_llm() override {
        try {
            return new LlamaCppLlm();
        } catch (...) {
            return nullptr;
        }
    }

    VlmBackend* create_vlm() override {
        try {
            return new LlamaCppVlm();
        } catch (...) {
            return nullptr;
        }
    }

    EmbeddingBackend* create_embedding() override {
        try {
            return new LlamaCppEmbedding();
        } catch (...) {
            return nullptr;
        }
    }
};

}  // namespace unirt::llama_plugin

unirt_PluginId unirt_plugin_id() {
    return unirt::build_config::kPluginIdLlamaCpp;
}

uint32_t unirt_plugin_abi_version() { return UNIRT_PLUGIN_ABI_VERSION; }

unirt_PluginTable* unirt_plugin_open() {
    return unirt::plugin_export::open_package<unirt::llama_plugin::LlamaCppPlugin>();
}

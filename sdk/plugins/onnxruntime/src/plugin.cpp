// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "plugin/plugin_export.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "build_config.h"
#include "embedding.h"
#include "logging.h"

namespace unirt::onnxruntime_plugin {

class OnnxRuntimePlugin final : public BackendPackage {
public:
  OnnxRuntimePlugin()
      : environment_(ORT_LOGGING_LEVEL_WARNING, "unirt-onnxruntime"),
        version_(Ort::GetVersionString()) {
    try {
      providers_ = Ort::GetAvailableProviders();
    } catch (const std::exception &error) {
      UNIRT_LOG_WARN("onnxruntime: cannot enumerate providers: {}",
                     error.what());
    }
  }

  const char *version() override { return version_.c_str(); }

  uint32_t modalities() override { return UNIRT_MODALITY_EMBEDDING; }

  int32_t get_device_list(const unirt_GetDeviceListInput *input,
                          unirt_GetDeviceListOutput *output) override {
    if (!input || !output)
      return UNIRT_ERROR_COMMON_INVALID_INPUT;
    *output = {};

    struct Device {
      const char *id;
      const char *name;
    };
    std::vector<Device> devices = {{"cpu", "ONNX Runtime CPU"}};
    if (has_provider("CoreMLExecutionProvider")) {
      devices.push_back({"coreml", "ONNX Runtime Core ML"});
    }
    if (devices.size() >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
    }

    auto **ids = static_cast<const char **>(
        std::calloc(devices.size(), sizeof(const char *)));
    auto **names = static_cast<const char **>(
        std::calloc(devices.size(), sizeof(const char *)));
    if (!ids || !names) {
      std::free(ids);
      std::free(names);
      return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
    }
    for (size_t index = 0; index < devices.size(); ++index) {
      ids[index] = devices[index].id;
      names[index] = devices[index].name;
    }
    output->device_ids = ids;
    output->device_names = names;
    output->device_count = static_cast<int32_t>(devices.size());
    return UNIRT_SUCCESS;
  }

  EmbeddingBackend *create_embedding() override {
    try {
      return new OnnxEmbedding(environment_);
    } catch (...) {
      return nullptr;
    }
  }

private:
  bool has_provider(const char *name) const {
    return std::find(providers_.begin(), providers_.end(), name) !=
           providers_.end();
  }

  Ort::Env environment_;
  std::string version_;
  std::vector<std::string> providers_;
};

} // namespace unirt::onnxruntime_plugin

unirt_PluginId unirt_plugin_id() { return unirt::build_config::kPluginIdOnnxRuntime; }

uint32_t unirt_plugin_abi_version() { return UNIRT_PLUGIN_ABI_VERSION; }

unirt_PluginTable* unirt_plugin_open() {
  return unirt::plugin_export::open_package<unirt::onnxruntime_plugin::OnnxRuntimePlugin>();
}

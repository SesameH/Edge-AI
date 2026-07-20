// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "plugin/embedding_backend.h"

namespace unirt::onnxruntime_plugin {

class OnnxEmbedding final : public EmbeddingBackend {
public:
  explicit OnnxEmbedding(Ort::Env &environment) : environment_(environment) {}

  int32_t create(const unirt_EmbeddingCreateInput *input) override;
  int32_t encode(const unirt_EmbeddingEncodeInput *input,
                 unirt_EmbeddingEncodeOutput *output) override;
  int32_t get_runtime_stats(unirt_EmbeddingRuntimeStats *output) override;

private:
  enum class InputKind {
    input_ids,
    attention_mask,
    token_type_ids,
    position_ids
  };

  struct InputSpec {
    std::string name;
    InputKind kind;
    ONNXTensorElementDataType element_type;
    std::vector<int64_t> shape;
  };

  bool shape_accepts(const InputSpec &spec, int32_t batch,
                     int32_t sequence) const noexcept;
  int32_t select_output(const char *requested_name);
  int32_t pool_output(const Ort::Value &value,
                      const unirt_EmbeddingEncodeInput *input,
                      unirt_EmbeddingEncodeOutput *output);

  Ort::Env &environment_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<InputSpec> inputs_;
  std::string output_name_;
  unirt_EmbeddingPooling pooling_ = UNIRT_EMBEDDING_POOLING_MODEL_DEFAULT;
  bool normalize_ = false;
  int64_t model_bytes_ = -1;
  std::string device_name_ = "ONNX Runtime CPU";
};

} // namespace unirt::onnxruntime_plugin

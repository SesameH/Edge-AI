// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>

#include "logging.h"

namespace unirt::onnxruntime_plugin {
namespace {

bool supported_float_type(ONNXTensorElementDataType type) noexcept {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

bool supported_integer_type(ONNXTensorElementDataType type) noexcept {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

bool provider_available(const char *wanted) {
  const auto providers = Ort::GetAvailableProviders();
  return std::find(providers.begin(), providers.end(), wanted) !=
         providers.end();
}

int output_score(const std::string &name, size_t rank) {
  static const std::unordered_map<std::string, int> priorities = {
      {"sentence_embedding", 120}, {"sentence_embeddings", 119},
      {"text_embeds", 115},        {"pooler_output", 110},
      {"last_hidden_state", 90},   {"token_embeddings", 85},
  };
  const auto found = priorities.find(name);
  if (found != priorities.end())
    return found->second;
  return rank == 2 ? 60 : 50;
}

template <typename Source>
void copy_as_float(const Source *source, size_t count,
                   std::vector<float> &destination) {
  destination.resize(count);
  std::transform(source, source + count, destination.begin(),
                 [](const Source &value) { return static_cast<float>(value); });
}

bool checked_product(size_t left, size_t right, size_t &result) noexcept {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
    return false;
  result = left * right;
  return true;
}

} // namespace

bool OnnxEmbedding::shape_accepts(const InputSpec &spec, int32_t batch,
                                  int32_t sequence) const noexcept {
  if (spec.shape.size() != 2)
    return false;
  return (spec.shape[0] <= 0 || spec.shape[0] == batch) &&
         (spec.shape[1] <= 0 || spec.shape[1] == sequence);
}

int32_t OnnxEmbedding::select_output(const char *requested_name) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t count = session_->GetOutputCount();
  int best_score = std::numeric_limits<int>::min();
  std::string best_name;
  bool requested_seen = false;

  for (size_t index = 0; index < count; ++index) {
    auto allocated_name = session_->GetOutputNameAllocated(index, allocator);
    const std::string name = allocated_name ? allocated_name.get() : "";
    if (requested_name && name == requested_name)
      requested_seen = true;

    const Ort::TypeInfo type_info = session_->GetOutputTypeInfo(index);
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR)
      continue;
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    const auto shape = tensor_info.GetShape();
    if ((shape.size() != 2 && shape.size() != 3) ||
        !supported_float_type(tensor_info.GetElementType())) {
      continue;
    }

    if (requested_name) {
      if (name == requested_name) {
        output_name_ = name;
        return UNIRT_SUCCESS;
      }
      continue;
    }
    const int score = output_score(name, shape.size());
    if (score > best_score) {
      best_score = score;
      best_name = name;
    }
  }

  if (requested_name) {
    UNIRT_LOG_ERROR("onnxruntime: requested output '{}' {}", requested_name,
                    requested_seen ? "is not a float rank-2/rank-3 tensor"
                                   : "does not exist");
    return requested_seen ? UNIRT_ERROR_COMMON_MODEL_INVALID
                          : UNIRT_ERROR_COMMON_INVALID_INPUT;
  }
  if (best_name.empty()) {
    UNIRT_LOG_ERROR("onnxruntime: graph has no poolable float output");
    return UNIRT_ERROR_COMMON_MODEL_INVALID;
  }
  output_name_ = std::move(best_name);
  return UNIRT_SUCCESS;
}

int32_t OnnxEmbedding::create(const unirt_EmbeddingCreateInput *input) {
  namespace fs = std::filesystem;
  if (!input || !input->model_path || !input->model_path[0]) {
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  }
  const fs::path model_path = fs::u8path(input->model_path);
  std::error_code error;
  if (!fs::is_regular_file(model_path, error)) {
    UNIRT_LOG_ERROR("onnxruntime: model file does not exist: {}",
                    input->model_path);
    return UNIRT_ERROR_COMMON_FILE_NOT_FOUND;
  }

  const std::string device = input->device_id ? input->device_id : "cpu";
  if (device != "cpu" && device != "coreml") {
    UNIRT_LOG_ERROR("onnxruntime: unknown device id '{}'", device);
    return UNIRT_ERROR_COMMON_INVALID_DEVICE;
  }

  try {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (device == "coreml") {
      if (!provider_available("CoreMLExecutionProvider")) {
        UNIRT_LOG_ERROR(
            "onnxruntime: Core ML execution provider is unavailable");
        return UNIRT_ERROR_COMMON_NOT_SUPPORTED;
      }
      options.AppendExecutionProvider(
          "CoreML", {{"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"}});
      device_name_ = "ONNX Runtime Core ML";
    } else {
      device_name_ = "ONNX Runtime CPU";
    }

    session_ = std::make_unique<Ort::Session>(environment_, model_path.c_str(),
                                              options);
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = session_->GetInputCount();
    inputs_.clear();
    inputs_.reserve(input_count);
    bool found_input_ids = false;

    for (size_t index = 0; index < input_count; ++index) {
      auto allocated_name = session_->GetInputNameAllocated(index, allocator);
      const std::string name = allocated_name ? allocated_name.get() : "";
      InputKind kind;
      if (name == "input_ids") {
        kind = InputKind::input_ids;
        found_input_ids = true;
      } else if (name == "attention_mask") {
        kind = InputKind::attention_mask;
      } else if (name == "token_type_ids") {
        kind = InputKind::token_type_ids;
      } else if (name == "position_ids") {
        kind = InputKind::position_ids;
      } else {
        UNIRT_LOG_ERROR("onnxruntime: unsupported required graph input '{}'",
                        name);
        session_.reset();
        return UNIRT_ERROR_COMMON_MODEL_INVALID;
      }

      const Ort::TypeInfo type_info = session_->GetInputTypeInfo(index);
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        session_.reset();
        return UNIRT_ERROR_COMMON_MODEL_INVALID;
      }
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      const auto shape = tensor_info.GetShape();
      if (shape.size() != 2 ||
          !supported_integer_type(tensor_info.GetElementType())) {
        UNIRT_LOG_ERROR(
            "onnxruntime: input '{}' must be a rank-2 int32/int64 tensor",
            name);
        session_.reset();
        return UNIRT_ERROR_COMMON_MODEL_INVALID;
      }
      inputs_.push_back({name, kind, tensor_info.GetElementType(), shape});
    }
    if (!found_input_ids) {
      UNIRT_LOG_ERROR("onnxruntime: graph does not declare input_ids");
      session_.reset();
      return UNIRT_ERROR_COMMON_MODEL_INVALID;
    }

    const int32_t selected = select_output(
        input->output_name && input->output_name[0] ? input->output_name
                                                    : nullptr);
    if (selected != UNIRT_SUCCESS) {
      session_.reset();
      return selected;
    }

    pooling_ = input->pooling;
    normalize_ = input->normalize;
    uintmax_t bytes = fs::file_size(model_path, error);
    if (!error) {
      for (const auto &suffix : {std::string("_data"), std::string(".data")}) {
        const fs::path external = model_path.parent_path() /
                                  (model_path.filename().string() + suffix);
        std::error_code external_error;
        if (!fs::is_regular_file(external, external_error))
          continue;
        const uintmax_t external_bytes =
            fs::file_size(external, external_error);
        if (external_error ||
            bytes > std::numeric_limits<uintmax_t>::max() - external_bytes) {
          error = std::make_error_code(std::errc::value_too_large);
          break;
        }
        bytes += external_bytes;
      }
    }
    model_bytes_ = !error && bytes <= static_cast<uintmax_t>(
                                          std::numeric_limits<int64_t>::max())
                       ? static_cast<int64_t>(bytes)
                       : -1;
    UNIRT_LOG_INFO("onnxruntime: loaded {} on {}, output '{}'",
                   input->model_path, device_name_, output_name_);
    return UNIRT_SUCCESS;
  } catch (const Ort::Exception &exception) {
    UNIRT_LOG_ERROR("onnxruntime: model load failed: {}", exception.what());
    session_.reset();
    return UNIRT_ERROR_COMMON_MODEL_LOAD;
  } catch (const std::bad_alloc &) {
    session_.reset();
    return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
  } catch (const std::exception &exception) {
    UNIRT_LOG_ERROR("onnxruntime: model load failed: {}", exception.what());
    session_.reset();
    return UNIRT_ERROR_COMMON_MODEL_LOAD;
  }
}

int32_t OnnxEmbedding::encode(const unirt_EmbeddingEncodeInput *input,
                              unirt_EmbeddingEncodeOutput *output) {
  if (!session_ || !input || !output || !input->input_ids ||
      input->batch_size <= 0 || input->sequence_length <= 0) {
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  }
  *output = {};
  const size_t batch = static_cast<size_t>(input->batch_size);
  const size_t sequence = static_cast<size_t>(input->sequence_length);
  size_t element_count = 0;
  if (!checked_product(batch, sequence, element_count)) {
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  }

  for (size_t index = 0; index < element_count; ++index) {
    if (input->input_ids[index] < 0 ||
        (input->attention_mask && input->attention_mask[index] != 0 &&
         input->attention_mask[index] != 1) ||
        (input->token_type_ids && input->token_type_ids[index] < 0)) {
      return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
  }
  for (const auto &spec : inputs_) {
    if (!shape_accepts(spec, input->batch_size, input->sequence_length)) {
      UNIRT_LOG_ERROR("onnxruntime: batch shape [{},{}] conflicts with static "
                      "input '{}' shape",
                      input->batch_size, input->sequence_length, spec.name);
      return UNIRT_ERROR_COMMON_INVALID_INPUT;
    }
  }

  try {
    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t dimensions[] = {input->batch_size, input->sequence_length};
    std::vector<std::vector<int64_t>> storage64;
    std::vector<std::vector<int32_t>> storage32;
    std::vector<const char *> names;
    std::vector<Ort::Value> values;
    storage64.reserve(inputs_.size());
    storage32.reserve(inputs_.size());
    names.reserve(inputs_.size());
    values.reserve(inputs_.size());

    for (const auto &spec : inputs_) {
      std::vector<int64_t> raw(element_count);
      switch (spec.kind) {
      case InputKind::input_ids:
        std::copy(input->input_ids, input->input_ids + element_count,
                  raw.begin());
        break;
      case InputKind::attention_mask:
        if (input->attention_mask) {
          std::copy(input->attention_mask,
                    input->attention_mask + element_count, raw.begin());
        } else {
          std::fill(raw.begin(), raw.end(), 1);
        }
        break;
      case InputKind::token_type_ids:
        if (input->token_type_ids) {
          std::copy(input->token_type_ids,
                    input->token_type_ids + element_count, raw.begin());
        }
        break;
      case InputKind::position_ids:
        for (size_t row = 0; row < batch; ++row) {
          for (size_t column = 0; column < sequence; ++column) {
            raw[row * sequence + column] = static_cast<int64_t>(column);
          }
        }
        break;
      }

      names.push_back(spec.name.c_str());
      if (spec.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        storage64.push_back(std::move(raw));
        auto &stored = storage64.back();
        values.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory, stored.data(), stored.size(), dimensions, 2));
      } else {
        std::vector<int32_t> converted;
        converted.reserve(raw.size());
        for (int64_t value : raw) {
          if (value > std::numeric_limits<int32_t>::max()) {
            return UNIRT_ERROR_COMMON_INVALID_INPUT;
          }
          converted.push_back(static_cast<int32_t>(value));
        }
        storage32.push_back(std::move(converted));
        auto &stored = storage32.back();
        values.emplace_back(Ort::Value::CreateTensor<int32_t>(
            memory, stored.data(), stored.size(), dimensions, 2));
      }
    }

    const char *output_name = output_name_.c_str();
    auto results = session_->Run(Ort::RunOptions{nullptr}, names.data(),
                                 values.data(), values.size(), &output_name, 1);
    if (results.size() != 1)
      return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
    return pool_output(results.front(), input, output);
  } catch (const Ort::Exception &exception) {
    UNIRT_LOG_ERROR("onnxruntime: inference failed: {}", exception.what());
    return UNIRT_ERROR_EMBEDDING_INFERENCE_FAILED;
  } catch (const std::bad_alloc &) {
    return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
  } catch (const std::exception &exception) {
    UNIRT_LOG_ERROR("onnxruntime: inference failed: {}", exception.what());
    return UNIRT_ERROR_EMBEDDING_INFERENCE_FAILED;
  }
}

int32_t OnnxEmbedding::pool_output(const Ort::Value &value,
                                   const unirt_EmbeddingEncodeInput *input,
                                   unirt_EmbeddingEncodeOutput *output) {
  if (!value.IsTensor())
    return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  if (shape.size() != 2 && shape.size() != 3) {
    return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
  }
  if (shape[0] != input->batch_size || shape.back() <= 0 ||
      shape.back() > std::numeric_limits<int32_t>::max()) {
    return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
  }
  if (shape.size() == 3 && shape[1] != input->sequence_length) {
    return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
  }

  const size_t source_count = info.GetElementCount();
  std::vector<float> source;
  switch (info.GetElementType()) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    copy_as_float(value.GetTensorData<float>(), source_count, source);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    copy_as_float(value.GetTensorData<double>(), source_count, source);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    copy_as_float(value.GetTensorData<Ort::Float16_t>(), source_count, source);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    copy_as_float(value.GetTensorData<Ort::BFloat16_t>(), source_count, source);
    break;
  default:
    return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
  }

  const size_t batch = static_cast<size_t>(input->batch_size);
  const size_t sequence = static_cast<size_t>(input->sequence_length);
  const size_t hidden = static_cast<size_t>(shape.back());
  size_t pooled_count = 0;
  if (!checked_product(batch, hidden, pooled_count) ||
      pooled_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
  }
  size_t expected_source = pooled_count;
  if (shape.size() == 3 &&
      !checked_product(expected_source, sequence, expected_source)) {
    return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
  }
  if (source_count != expected_source)
    return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;

  std::vector<float> pooled(pooled_count, 0.0f);
  if (shape.size() == 2) {
    pooled = std::move(source);
  } else {
    const auto selected_pooling =
        pooling_ == UNIRT_EMBEDDING_POOLING_MODEL_DEFAULT
            ? UNIRT_EMBEDDING_POOLING_MEAN
            : pooling_;
    for (size_t row = 0; row < batch; ++row) {
      size_t token_index = 0;
      if (selected_pooling == UNIRT_EMBEDDING_POOLING_LAST_TOKEN) {
        bool found = false;
        for (size_t column = sequence; column > 0; --column) {
          if (!input->attention_mask ||
              input->attention_mask[row * sequence + column - 1] != 0) {
            token_index = column - 1;
            found = true;
            break;
          }
        }
        if (!found)
          return UNIRT_ERROR_COMMON_INVALID_INPUT;
      }

      if (selected_pooling == UNIRT_EMBEDDING_POOLING_CLS ||
          selected_pooling == UNIRT_EMBEDDING_POOLING_LAST_TOKEN) {
        const size_t source_offset = (row * sequence + token_index) * hidden;
        std::copy_n(source.data() + source_offset, hidden,
                    pooled.data() + row * hidden);
        continue;
      }

      size_t visible = 0;
      for (size_t column = 0; column < sequence; ++column) {
        if (input->attention_mask &&
            input->attention_mask[row * sequence + column] == 0) {
          continue;
        }
        ++visible;
        const size_t source_offset = (row * sequence + column) * hidden;
        for (size_t dimension = 0; dimension < hidden; ++dimension) {
          pooled[row * hidden + dimension] += source[source_offset + dimension];
        }
      }
      if (visible == 0)
        return UNIRT_ERROR_COMMON_INVALID_INPUT;
      const float scale = 1.0f / static_cast<float>(visible);
      for (size_t dimension = 0; dimension < hidden; ++dimension) {
        pooled[row * hidden + dimension] *= scale;
      }
    }
  }

  if (normalize_) {
    for (size_t row = 0; row < batch; ++row) {
      double squared_norm = 0.0;
      for (size_t dimension = 0; dimension < hidden; ++dimension) {
        const double component = pooled[row * hidden + dimension];
        squared_norm += component * component;
      }
      if (squared_norm <= 0.0 || !std::isfinite(squared_norm)) {
        return UNIRT_ERROR_EMBEDDING_OUTPUT_INVALID;
      }
      const float inverse_norm =
          static_cast<float>(1.0 / std::sqrt(squared_norm));
      for (size_t dimension = 0; dimension < hidden; ++dimension) {
        pooled[row * hidden + dimension] *= inverse_norm;
      }
    }
  }

  float *owned =
      static_cast<float *>(std::malloc(pooled_count * sizeof(float)));
  if (!owned)
    return UNIRT_ERROR_COMMON_MEMORY_ALLOCATION;
  std::memcpy(owned, pooled.data(), pooled_count * sizeof(float));
  output->embeddings = owned;
  output->embedding_count = input->batch_size;
  output->embedding_dimension = static_cast<int32_t>(hidden);
  return UNIRT_SUCCESS;
}

int32_t OnnxEmbedding::get_runtime_stats(unirt_EmbeddingRuntimeStats *output) {
  if (!output)
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  output->model_bytes = model_bytes_;
  output->device_peak_bytes = -1;
  output->device_name = device_name_.c_str();
  return UNIRT_SUCCESS;
}

} // namespace unirt::onnxruntime_plugin

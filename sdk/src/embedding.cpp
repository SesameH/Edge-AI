// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// unirt_embedding_* C entrypoints: strict shape/ownership validation and
// forwarding through the same exception wall used by generation backends.

#include <cstring>
#include <limits>

#include "bridge_support.h"
#include "plugin/embedding_backend.h"

using namespace unirt;
using namespace unirt::bridge;

namespace {

bool pooling_sane(unirt_EmbeddingPooling pooling) noexcept {
  return pooling >= UNIRT_EMBEDDING_POOLING_MODEL_DEFAULT &&
         pooling <= UNIRT_EMBEDDING_POOLING_LAST_TOKEN;
}

bool encode_input_sane(const unirt_EmbeddingEncodeInput *input) noexcept {
  if (!input || !input->input_ids || input->batch_size <= 0 ||
      input->sequence_length <= 0) {
    return false;
  }
  const auto batch = static_cast<size_t>(input->batch_size);
  const auto width = static_cast<size_t>(input->sequence_length);
  return batch <= std::numeric_limits<size_t>::max() / width;
}

bool rerank_input_sane(const unirt_EmbeddingRerankInput *input) noexcept {
  if (!input || !input->query_utf8 || !input->query_utf8[0] ||
      !input->documents_utf8 || input->document_count <= 0) {
    return false;
  }
  for (int32_t i = 0; i < input->document_count; ++i) {
    if (!input->documents_utf8[i] || !input->documents_utf8[i][0]) return false;
  }
  return true;
}

} // namespace

int32_t unirt_embedding_create(const unirt_EmbeddingCreateInput *input,
                               unirt_Embedding **out_handle) {
  if (!out_handle)
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  *out_handle = nullptr;
  if (!input || !input->model_path || !input->model_path[0] ||
      !input->plugin_id || !input->plugin_id[0] ||
      !pooling_sane(input->pooling)) {
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  }
  UNIRT_LOG_TRACE("{}", input);
  return open_backend<EmbeddingBackend>("embedding.create", input, out_handle);
}

int32_t unirt_embedding_destroy(unirt_Embedding *handle) {
  UNIRT_LOG_TRACE("embedding.destroy");
  return close_backend<EmbeddingBackend>("embedding.destroy", handle);
}

int32_t unirt_embedding_encode(unirt_Embedding *handle,
                               const unirt_EmbeddingEncodeInput *input,
                               unirt_EmbeddingEncodeOutput *output) {
  if (!output)
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  *output = {};
  if (!encode_input_sane(input))
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  return with_backend<EmbeddingBackend>(
      "embedding.encode", handle, UNIRT_ERROR_EMBEDDING_INFERENCE_FAILED,
      [&](EmbeddingBackend &model) { return model.encode(input, output); });
}

int32_t unirt_embedding_get_runtime_stats(unirt_Embedding *handle,
                                          unirt_EmbeddingRuntimeStats *output) {
  if (!output)
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  std::memset(output, 0, sizeof(*output));
  output->model_bytes = -1;
  output->device_peak_bytes = -1;
  return with_backend<EmbeddingBackend>(
      "embedding.get_runtime_stats", handle, UNIRT_ERROR_COMMON_UNKNOWN,
      [&](EmbeddingBackend &model) {
        const int32_t rc = model.get_runtime_stats(output);
        output->process_rss_bytes = resident_set_bytes();
        return rc;
      });
}

int32_t unirt_embedding_rerank(unirt_Embedding *handle,
                               const unirt_EmbeddingRerankInput *input,
                               unirt_EmbeddingRerankOutput *output) {
  if (!output)
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  *output = {};
  if (!rerank_input_sane(input))
    return UNIRT_ERROR_COMMON_INVALID_INPUT;
  UNIRT_LOG_TRACE("embedding.rerank");
  return with_backend<EmbeddingBackend>(
      "embedding.rerank", handle, UNIRT_ERROR_EMBEDDING_INFERENCE_FAILED,
      [&](EmbeddingBackend &model) { return model.rerank(input, output); });
}

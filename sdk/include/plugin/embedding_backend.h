// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "unirt.h"

namespace unirt {

/** Native encoder contract implemented by embedding-capable plugins. */
class EmbeddingBackend {
public:
  virtual ~EmbeddingBackend() = default;

  /** Load one encoder graph. Called exactly once before encode. */
  virtual int32_t create(const unirt_EmbeddingCreateInput *input) = 0;

  /** Execute a rectangular token batch and return caller-owned float32 rows. */
  virtual int32_t encode(const unirt_EmbeddingEncodeInput *input,
                         unirt_EmbeddingEncodeOutput *output) = 0;

  /** Report model/device memory where the backend can measure it. */
  virtual int32_t get_runtime_stats(unirt_EmbeddingRuntimeStats *) {
    return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
  }

  /** Score a query against candidate documents with a classifier/rerank
   *  head. Default: unsupported, for backends/models without one. */
  virtual int32_t rerank(const unirt_EmbeddingRerankInput *,
                         unirt_EmbeddingRerankOutput *) {
    return UNIRT_ERROR_COMMON_PARAM_NOT_SUPPORTED;
  }
};

} // namespace unirt

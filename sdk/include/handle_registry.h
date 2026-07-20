// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>

#include "plugin/embedding_backend.h"
#include "plugin/llm_backend.h"
#include "plugin/vlm_backend.h"

namespace unirt {

// Public handles are opaque, but they are NOT backend pointers: each is an
// encoded {slot index, generation} pair minted by this registry. Every C
// entrypoint decodes and validates before any dereference, so NULL, random,
// wrong-modality, and stale handles all fail cleanly — including the ABA
// case where a freed backend's address is reused by a new one (the
// generation stamped into the old handle no longer matches its slot).
// The C API documents operations on one handle as non-concurrent; this
// registry protects validation/destruction and runtime teardown, not
// concurrent inference on the same model.
unirt_LLM* register_llm_handle(LlmBackend* backend);
unirt_VLM* register_vlm_handle(VlmBackend* backend);
unirt_Embedding* register_embedding_handle(EmbeddingBackend* backend);
LlmBackend* find_llm_handle(const unirt_LLM* handle) noexcept;
VlmBackend* find_vlm_handle(const unirt_VLM* handle) noexcept;
EmbeddingBackend* find_embedding_handle(const unirt_Embedding* handle) noexcept;
LlmBackend* remove_llm_handle(unirt_LLM* handle) noexcept;
VlmBackend* remove_vlm_handle(unirt_VLM* handle) noexcept;
EmbeddingBackend* remove_embedding_handle(unirt_Embedding* handle) noexcept;
std::size_t active_handle_count() noexcept;

}  // namespace unirt

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#include "handle_registry.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace unirt {
namespace {

// Handle words pack {generation:32 | slot+1:32}; slots are never handed out
// as zero, so a NULL pointer can never decode to a live slot.
static_assert(sizeof(void*) >= 8, "handle encoding assumes 64-bit pointers");

enum class Kind : uint8_t { llm, vlm, embedding };

struct Slot {
    void*    backend    = nullptr;  // null while the slot is free
    uint32_t generation = 0;        // bumped on every release; stale handles mismatch
    Kind     kind       = Kind::llm;
};

std::mutex            table_mutex;
std::vector<Slot>     slots;
std::vector<uint32_t> free_slots;

void* encode(uint32_t index, uint32_t generation) noexcept {
    return reinterpret_cast<void*>(
        (static_cast<uint64_t>(generation) << 32) | static_cast<uint64_t>(index + 1));
}

void* mint(void* backend, Kind kind) {
    if (!backend) return nullptr;
    std::lock_guard<std::mutex> lock(table_mutex);
    uint32_t index;
    if (!free_slots.empty()) {
        index = free_slots.back();
        free_slots.pop_back();
    } else {
        index = static_cast<uint32_t>(slots.size());
        slots.emplace_back();
    }
    Slot& slot   = slots[index];
    slot.backend = backend;
    slot.kind    = kind;
    if (slot.generation == 0) slot.generation = 1;
    return encode(index, slot.generation);
}

// Decode and validate under the lock; returns the slot's backend or null.
void* resolve(const void* handle, Kind kind, bool release) noexcept {
    const auto word = reinterpret_cast<uint64_t>(handle);
    const auto low  = static_cast<uint32_t>(word & 0xffffffffu);
    if (low == 0) return nullptr;
    const uint32_t index      = low - 1;
    const auto     generation = static_cast<uint32_t>(word >> 32);

    std::lock_guard<std::mutex> lock(table_mutex);
    if (index >= slots.size()) return nullptr;
    Slot& slot = slots[index];
    if (!slot.backend || slot.kind != kind || slot.generation != generation) return nullptr;

    void* backend = slot.backend;
    if (release) {
        slot.backend = nullptr;
        ++slot.generation;  // invalidates every copy of the old handle word
        free_slots.push_back(index);
    }
    return backend;
}

}  // namespace

unirt_LLM* register_llm_handle(LlmBackend* backend) {
    return static_cast<unirt_LLM*>(mint(backend, Kind::llm));
}

unirt_VLM* register_vlm_handle(VlmBackend* backend) {
    return static_cast<unirt_VLM*>(mint(backend, Kind::vlm));
}

unirt_Embedding* register_embedding_handle(EmbeddingBackend* backend) {
    return static_cast<unirt_Embedding*>(mint(backend, Kind::embedding));
}

LlmBackend* find_llm_handle(const unirt_LLM* handle) noexcept {
    return static_cast<LlmBackend*>(resolve(handle, Kind::llm, false));
}

VlmBackend* find_vlm_handle(const unirt_VLM* handle) noexcept {
    return static_cast<VlmBackend*>(resolve(handle, Kind::vlm, false));
}

EmbeddingBackend* find_embedding_handle(const unirt_Embedding* handle) noexcept {
    return static_cast<EmbeddingBackend*>(resolve(handle, Kind::embedding, false));
}

LlmBackend* remove_llm_handle(unirt_LLM* handle) noexcept {
    return static_cast<LlmBackend*>(resolve(handle, Kind::llm, true));
}

VlmBackend* remove_vlm_handle(unirt_VLM* handle) noexcept {
    return static_cast<VlmBackend*>(resolve(handle, Kind::vlm, true));
}

EmbeddingBackend* remove_embedding_handle(unirt_Embedding* handle) noexcept {
    return static_cast<EmbeddingBackend*>(resolve(handle, Kind::embedding, true));
}

std::size_t active_handle_count() noexcept {
    std::lock_guard<std::mutex> lock(table_mutex);
    std::size_t live = 0;
    for (const Slot& slot : slots) {
        if (slot.backend) ++live;
    }
    return live;
}

}  // namespace unirt

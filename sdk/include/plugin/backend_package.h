// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include "embedding_backend.h"
#include "llm_backend.h"
#include "plugin_abi.h"
#include "unirt.h"
#include "vlm_backend.h"

namespace unirt {

/**
 * C++ convenience base for plugin authors. NOT the ABI — the boundary
 * contract is the C tables in plugin_abi.h; plugin_export.h turns an
 * implementation of these classes into those tables. The bridge wraps the
 * tables back into these interfaces (plugin_adapters.h), so both sides
 * program against C++ while only C crosses the boundary.
 *
 * A backend package exposes identity, device enumeration, and factories for
 * the modality interfaces it supports (returning nullptr for the ones it
 * does not).
 */
class BackendPackage {
   public:
    virtual ~BackendPackage() = default;

    virtual const char* version() { return "unknown"; }

    /** UNIRT_MODALITY_* bits for the model kinds this package can host.
     *  Declared statically so listings need not instantiate anything. */
    virtual uint32_t modalities() { return 0; }

    /** Enumerate compute devices this backend can run on. The default
     *  reports an empty list. */
    virtual int32_t get_device_list(const unirt_GetDeviceListInput*, unirt_GetDeviceListOutput* output) {
        if (!output) return UNIRT_ERROR_COMMON_INVALID_INPUT;
        output->device_ids   = nullptr;
        output->device_names = nullptr;
        output->device_count = 0;
        return UNIRT_SUCCESS;
    }

    virtual LlmBackend* create_llm() { return nullptr; }
    virtual VlmBackend* create_vlm() { return nullptr; }
    virtual EmbeddingBackend* create_embedding() { return nullptr; }
};

}  // namespace unirt

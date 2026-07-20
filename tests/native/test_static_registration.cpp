// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// Static plugin registration: a backend linked into the binary (no dlopen)
// must join the registry via unirt_register_plugin and be reachable through
// the normal C API. This is the loading model iOS requires.

#include <cstdio>
#include <cstring>

#include "plugin/plugin_export.h"
#include "unirt.h"

namespace {

class StubPackage final : public unirt::BackendPackage {
   public:
    const char* version() override { return "stub/1"; }
    uint32_t    modalities() override { return UNIRT_MODALITY_LLM; }
};

unirt_PluginId stub_id() { return "stub_static"; }

unirt_PluginTable* stub_open() {
    return unirt::plugin_export::open_package<StubPackage>();
}

#define REQUIRE(condition)                                              \
    do {                                                                \
        if (!(condition)) {                                             \
            std::fprintf(stderr, "FAILED: %s (line %d)\n", #condition, __LINE__); \
            return 1;                                                   \
        }                                                               \
    } while (0)

}  // namespace

int main() {
    // Registration is legal before init; the backend must survive it.
    REQUIRE(unirt_register_plugin(stub_id, stub_open) == UNIRT_SUCCESS);
    REQUIRE(unirt_register_plugin(nullptr, stub_open) == UNIRT_ERROR_COMMON_INVALID_INPUT);

    REQUIRE(unirt_init() == UNIRT_SUCCESS);

    unirt_GetPluginListOutput listing{};
    REQUIRE(unirt_get_plugin_list(&listing) == UNIRT_SUCCESS);
    bool found = false;
    for (int32_t i = 0; i < listing.plugin_count; ++i) {
        if (std::strcmp(listing.plugin_ids[i], "stub_static") == 0) found = true;
    }
    unirt_free(listing.plugin_ids);
    REQUIRE(found);

    uint32_t bits = 0;
    REQUIRE(unirt_get_plugin_modalities("stub_static", &bits) == UNIRT_SUCCESS);
    REQUIRE(bits == UNIRT_MODALITY_LLM);

    const char* reported = unirt_get_plugin_version("stub_static");
    REQUIRE(reported && std::strcmp(reported, "stub/1") == 0);

    REQUIRE(unirt_deinit() == UNIRT_SUCCESS);
    std::puts("static registration: OK");
    return 0;
}

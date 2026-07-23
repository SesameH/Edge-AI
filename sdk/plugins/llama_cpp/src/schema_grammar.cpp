// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// The one place the plugin reaches into llama.cpp's common library — for
// json_schema_to_grammar() only; everything else stays on the public
// llama.h/mtmd API. Isolated in its own TU so nlohmann/json.hpp isn't
// pulled into the (already large) llm/vlm translation units.

#include "schema_grammar.h"

#include <exception>

#include <nlohmann/json.hpp>

#include "json-schema-to-grammar.h"
#include "logging.h"
#include "unirt.h"

namespace unirt::llama_plugin {

std::string schema_to_gbnf(const char* schema_utf8, int32_t& error) {
    try {
        // force_gbnf: the result feeds llama_sampler_init_grammar, which
        // speaks GBNF only (never the llguidance dialect).
        return json_schema_to_grammar(nlohmann::ordered_json::parse(schema_utf8), true);
    } catch (const std::exception& e) {
        UNIRT_LOG_ERROR("llama_cpp: JSON Schema rejected: {}", e.what());
        error = UNIRT_ERROR_COMMON_INVALID_INPUT;
        return {};
    }
}

}  // namespace unirt::llama_plugin

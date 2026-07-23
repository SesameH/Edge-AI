// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <string>

namespace unirt::llama_plugin {

/** Compiles a JSON Schema (UTF-8 JSON text) into GBNF via llama.cpp's
 *  converter. Returns the grammar, or an empty string with `error` set to
 *  UNIRT_ERROR_COMMON_INVALID_INPUT when the schema doesn't parse or uses
 *  constructs the converter rejects. */
std::string schema_to_gbnf(const char* schema_utf8, int32_t& error);

}  // namespace unirt::llama_plugin

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace unirt::llama_plugin {

/**
 * Register the ggml backend modules installed beside this plugin.
 *
 * Only does anything in a UNIRT_GGML_BACKEND_DL build, where ggml's backends
 * are separate loadable modules rather than code linked into libggml. ggml's
 * own default search looks next to the *executable* and in the working
 * directory -- both wrong here, since the executable is whichever python3 or
 * host app loaded us. Idempotent, and safe to call from several threads.
 */
void load_ggml_backend_modules();

}  // namespace unirt::llama_plugin

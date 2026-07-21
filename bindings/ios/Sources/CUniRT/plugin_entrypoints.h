// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// A statically linked backend (the loading model iOS requires — see
// README.md) exports exactly these two C symbols, named identically for
// every backend per plugin/plugin_export.h; only one backend can be linked
// this way per binary. Declared here so Swift can pass them to
// UniRT.registerStaticPlugin without a bespoke bridging header.

#include "../../../../sdk/include/unirt.h"

extern unirt_PluginId unirt_plugin_id(void);
extern unirt_PluginTable* unirt_plugin_open(void);

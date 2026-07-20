# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

UniRT: a multi-backend on-device LLM/VLM/embedding inference SDK. One stable C ABI
(`sdk/include/unirt.h`) fronts interchangeable backend plugins; Python bindings, an
interactive chat example, and an OpenAI-compatible server sit on top. Derived from
Qualcomm GenieX (BSD-3-Clause) — see "License discipline" below.

## Build & test

```sh
# One-time prerequisites: Xcode CLT, Metal toolchain (xcodebuild -downloadComponent MetalToolchain), CMake.
# Rust is NOT required (the old Rust model-manager is gone).

git submodule update --init                       # third-party/llama.cpp, third-party/mlx

# MLX static lib (once, or after bumping the mlx submodule)
cmake -S third-party/mlx -B build-mlx -DCMAKE_BUILD_TYPE=Release \
      -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_INSTALL_PREFIX="$PWD/build-mlx/install"
cmake --build build-mlx -j8 && cmake --install build-mlx

# SDK + plugins (the normal dev loop)
cmake -S sdk -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cmake --install build --prefix sdk/pkg-unirt      # REQUIRED after every C++ change:
                                                  # the Python loader auto-discovers sdk/pkg-unirt/lib

# Tests (the behavior spec — keep them green)
python3 -m pytest tests/python                    # full suite
python3 -m pytest tests/python/test_smoke.py -k prefix -v   # single test

# Run it
PYTHONPATH=$PWD/bindings/python python3 examples/chat.py --backend mlx --model models/SmolLM2-135M-Instruct
PYTHONPATH=$PWD/bindings/python python3 -m unirt.server --model <path> --backend llama_cpp --port 8080
```

Test models live in `models/` (gitignored). Download commands are in README.md
("Getting started" step 4); tests skip when models are absent. Generator is
Make (no Ninja on this machine). `build/`, `build-mlx/`, `sdk/pkg-unirt/` are
all generated — never edit them.

## Architecture

Three layers, strictly separated:

1. **C ABI bridge** (`sdk/src/`, public contract in `sdk/include/unirt.h`).
   Every `unirt_*` entrypoint is: validate args → resolve the opaque handle →
   forward to the plugin behind a uniform exception wall. All of that
   machinery lives in `sdk/src/bridge_support.h` (`shielded`, `with_backend`,
   `open_backend`, `close_backend`, `HandleTraits<LlmBackend|VlmBackend|EmbeddingBackend>`,
   `run_generation`). A new C API is a few lines: validation + a lambda.
   - `registry.h/cpp`: `DynamicLibrary` (RAII dlopen) → `PluginSlot` (checks
     `unirt_plugin_abi_version()` before touching any vtable) → `PluginDirectory`
     (thread-safe; distinguishes "plugin never existed" from "exists but broken").
   - `handle_registry`: every public handle is validated against a live table,
     so stale/wrong-modality handles fail cleanly; `unirt_deinit` returns BUSY
     while handles are open.
   - `runtime.cpp`: init/deinit lifecycle, stderr log sink, ARM hwcap gate.
   - The bridge splices `StreamJoiner` (utils.h) between plugin token pieces
     and the caller's callback — plugins may emit partial UTF-8 freely.

2. **Plugins** (`sdk/plugins/<name>/`), one dynamic library each, exporting
   exactly `unirt_plugin_id()`, `unirt_plugin_abi_version()`,
   `unirt_plugin_open()`. Loaded at
   runtime from `<plugin root>/<name>/libunirt_plugin.dylib`.
   - **The boundary is pure C** (`plugin/plugin_abi.h`): function-pointer
     tables (`unirt_PluginTable` → `unirt_LlmTable`/`unirt_VlmTable`/
     `unirt_EmbeddingTable`), each with `struct_size` + `self` + `destroy`.
     No C++ types, vtables, or exceptions cross it, and `destroy(self)` frees
     in the allocating module. Plugin authors still write C++ classes
     (`BackendPackage`/`LlmBackend`/… in `plugin/backend_package.h`);
     `plugin/plugin_export.h` wraps them into tables (catching all
     exceptions), and the bridge's `plugin_adapters.h` wraps tables back
     into the same C++ interfaces — so `HandleTraits`/`with_backend` never
     see the C layer.
   - `llama_cpp`: GGUF via llama.cpp public API only (sampler chains, chat
     template, KV save/load, VLM via mtmd).
   - `mlx`: HF safetensors (Llama arch, dense or MLX-quantized) on Apple
     Silicon — own BPE tokenizer (`tokenizer.cpp`), own transformer graph
     (`model.cpp` on MLX C++ `fast::` ops), CPU-side sampler (`sampler.h`).
   - `onnxruntime`: embedding backend (`EmbeddingBackend`).
   - Both text backends implement **prefix cache** (a resent conversation
     transcript only evaluates the new suffix) and llama_cpp implements
     **context shifting** (evict-oldest on overflow). The plugin keeps a
     token transcript that must mirror KV contents exactly — in llama_cpp,
     `decode()` itself appends to `history_`; don't append at call sites.

3. **Python** (`bindings/python/unirt/`): ctypes FFI (`_ffi/`) → high-level
   `AutoModelForCausalLM` (`auto.py`, `modeling.py`) → `model_manager.py`
   (pure-Python, huggingface_hub-backed download/cache under
   `~/.cache/unirt/models`, `unirt.json` index) → `server.py` (stdlib-only
   OpenAI-compatible HTTP + SSE) and `cli.py`. Library discovery order:
   `UNIRT_LIB_PATH` env → wheel layout → `<repo>/sdk/pkg-unirt/lib`.

## Invariants that are easy to break

- **C ABI is frozen.** `unirt.h` struct layouts and signatures must stay
  byte-compatible with `bindings/python/unirt/_ffi/_types.py`. Change both
  together or neither; additions go at the end of structs.
- **Error codes** live in bands (`-10xx` runtime/args, `-11xx` hub, `-12xx`
  loading, `-13xx` LLM, `-14xx` VLM, `-15xx` embedding) and are mirrored as
  literals in `_ffi/_api.py`, `model_manager.py`, `auto.py`, and tests —
  renumbering means updating every mirror.
- **Handles are not pointers.** A public handle encodes {slot, generation}
  in `handle_registry.cpp`; stale handles fail even if the backend's memory
  is reused. Never hand a backend pointer to the caller.
- **Failure detail**: `bridge::fail(code, message)` records thread-local
  prose behind `unirt_last_error_message()`; `shielded()` clears it on entry
  and records every caught exception. Use `fail()` (not bare returns) for
  validation errors that merit an explanation; Python's `_check` appends the
  detail to `UniRTError`.
- **Plugin ABI stamp**: reshaping existing fields of the C tables in
  `plugin/plugin_abi.h` requires bumping `UNIRT_PLUGIN_ABI_VERSION` there.
  Appending fields at the end does NOT (that is what `struct_size` is for) —
  guard reads with `UNIRT_TABLE_HAS_FIELD` and keep `kRequiredTableBytes`
  at the offset of the first appended field (see `modalities`, the worked
  example: declared by `BackendPackage::modalities()`, stamped into the
  table, surfaced via `unirt_get_plugin_modalities`).
- **`n_past` semantics** in `unirt_GenerationConfig`: `0` = auto (prefix-match
  decides what KV survives), `>0` = explicit rewind to that point. `0` does
  NOT mean "clear"; `reset()` is the explicit clear.
- **Failure paths zero caller-owned output structs** (test_abi_guards enforces
  this): zero the output first, then validate.
- **Tests are the spec.** `tests/python/` encodes intended behavior
  (determinism of greedy, streaming == blocking text, stop-sequence
  non-leakage, CJK streaming reassembly, ABI guards). Run them after any
  change; the suite takes ~5 s.

## License discipline

Every file is original work under the Peter Huang BSD-3-Clause header; the
project began as a fork of Qualcomm GenieX but all derived code has since
been genuinely rewritten (verified by diffing against the GenieX sources —
remaining identical lines are ABI-dictated, not expressive). Keep it that
way: never copy code or prose (including comments and error messages) from
the GenieX repo into this codebase; if that ever becomes necessary, the
copied file must carry the upstream copyright notice per BSD-3-Clause.
`NOTICE` lists third-party dependency attributions — update it when adding
a dependency. Public API names that mirror HuggingFace conventions
(`AutoModelForCausalLM`, `GenerationConfig`, `TextIteratorStreamer`) are
deliberate compatibility surface, not derivation.

## Current gaps (do not assume these exist)

- Nothing has ever been committed to git — the working tree is the only copy.
- MLX backend: no KV-cache save/load, no VLM, no grammar-constrained decoding.
- No Go CLI / Android binding. Linux/Windows builds unverified since the fork.

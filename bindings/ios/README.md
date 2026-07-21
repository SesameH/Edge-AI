# UniRT iOS binding

Swift wrapper (`UniRTKit`) over the UniRT C API, plus a `CUniRT` module that
exposes `sdk/include/unirt.h` to Swift. Text generation (llama_cpp / GGUF)
only for now; VLM and embeddings follow the same pattern when needed.

Unlike Android, there is no JNI-style glue layer: Swift calls the C ABI
directly. The one iOS-specific wrinkle is plugin loading — iOS forbids
`dlopen` of arbitrary paths, so plugins cannot be discovered from a
directory scan the way `llama_cpp` is on macOS/Linux/Windows. Instead the
app links the plugin as a **static library** and joins it in-process with
`unirt_register_plugin()` before `unirt_init()`. `sdk/CMakeLists.txt`
already builds this way when `UNIRT_BUILD_SHARED=OFF` (the same
configuration the `ios` CI job uses).

## Build the native static libraries

```sh
git submodule update --init --depth 1 third-party/llama.cpp

cmake -S sdk -B build-ios -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
  -DUNIRT_BUILD_SHARED=OFF -DBUILD_SHARED_LIBS=OFF \
  -DUNIRT_PYTHON_TESTS=OFF \
  -DUNIRT_PLUGIN_MLX=OFF -DUNIRT_PLUGIN_ONNXRUNTIME=OFF
cmake --build build-ios -j8
```

(Swap `-DCMAKE_OSX_SYSROOT=iphoneos` for `iphonesimulator` for a simulator
build — that's what CI does to run the native tests under `simctl`; see
`.github/workflows/ci.yml`'s `ios` job.)

This produces, notably:

- `build-ios/src/libunirt.a` — the C ABI bridge (static)
- `build-ios/plugins/llama_cpp/src/libunirt_llama_cpp.a` — the llama_cpp
  backend (static), exporting `unirt_plugin_id()` / `unirt_plugin_open()`
- `build-ios/bin/libllama.dylib`, `libmtmd.dylib`, `libggml.dylib`,
  `libggml-cpu.dylib`, `libggml-base.dylib` — llama.cpp's **own** libraries,
  which `libunirt_llama_cpp.a` links against. `UNIRT_BUILD_SHARED=OFF` only
  controls UniRT's own targets (`libunirt`/`libunirt_llama_cpp`, so the app
  can skip the dlopen-based plugin scan); llama.cpp's CMake build produces
  its usual shared `libggml*`/`libllama`/`libmtmd`, independent of that flag.
  They're built with `@rpath`-relative install names (`otool -L` on
  `libunirt_llama_cpp.a`'s dependents confirms this), so a real app embeds
  them normally — no different from any other vendored dynamic framework.

## Wire it into an Xcode project

1. Add `bindings/ios` as a local Swift package dependency (Xcode: File →
   Add Package Dependencies → Add Local...) and link `UniRTKit` to your app
   target.
2. Add `libunirt.a` and `libunirt_llama_cpp.a` to the app target's "Link
   Binary With Libraries" build phase (static; no signing needed).
3. Add the five `libggml*`/`libllama`/`libmtmd` dylibs to "Frameworks,
   Libraries, and Embedded Content" set to **Embed & Sign** — they're real
   runtime dependencies of `libunirt_llama_cpp.a`, not build-time-only, and
   need to ship inside the app bundle for a device build.
4. Before calling `UniRT.start()`, register the static plugin(s) you linked:

```swift
import UniRTKit

try UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)
try UniRT.start()
```

## Use

```swift
import UniRTKit

try UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)
try UniRT.start()

let session = try await UniRT.createLlmSession(
    modelPath: "/path/to/SmolLM2-135M-Instruct-Q8_0.gguf")
let prompt = try await session.applyChatTemplate([.user("What is the capital of France?")])
for try await piece in session.stream(prompt: prompt) {
    print(piece, terminator: "")     // cancel the enclosing Task to stop decoding
}

try UniRT.stop()
```

`LlmSession` is a Swift `actor`: the native handle is single-threaded by
contract, and actor isolation confines every native call without extra
locking, mirroring the Kotlin binding's dedicated-dispatcher approach.

Models ship however the app prefers (bundled resource, downloaded at first
run); pass an absolute filesystem path — the sandbox means that's usually
somewhere under `FileManager.default.urls(for: .documentDirectory, ...)`
or `Bundle.main`.

## Run the integration test

`Tests/UniRTKitTests/InferenceSmokeTests.swift` is the Swift-layer
counterpart to `tests/native/test_inference_smoke.cpp`: it registers the
real llama_cpp plugin, loads a GGUF model, applies the chat template, and
runs both blocking and streaming generation. `swift test`/`xcodebuild test`
don't know how to find `libunirt`/`libunirt_llama_cpp`/llama.cpp's dylibs on
their own (`Package.swift` deliberately has no linker flags baked in, so the
package stays usable by apps that supply the libraries their own way), so
point the linker and the runtime loader at the `build-ios` tree explicitly,
and give the test the model path through an environment variable (it
`XCTSkip`s without one):

```sh
ROOT="$PWD"   # repo root
export TEST_RUNNER_UNIRT_TEST_MODEL_PATH="$ROOT/models/SmolLM2-135M-Instruct-Q8_0.gguf"
xcodebuild test -scheme UniRTKit -destination "id=$SIM_UDID" \
  LIBRARY_SEARCH_PATHS="\$(inherited) $ROOT/build-ios/src $ROOT/build-ios/plugins/llama_cpp/src $ROOT/build-ios/bin" \
  OTHER_LDFLAGS="\$(inherited) -lunirt -lunirt_llama_cpp -lmtmd -lllama -lggml -lggml-cpu -lggml-base -lc++" \
  LD_RUNPATH_SEARCH_PATHS="\$(inherited) $ROOT/build-ios/bin"
```

(`TEST_RUNNER_`-prefixed variables are xcodebuild's mechanism for passing
environment into the test process; `$SIM_UDID` is whatever simulator you
booted — reuse the one from the `ios` CI job's "Boot a simulator" step.)
This exact recipe (same flags, same library layout) is verified end to end
against a real model on a static macOS build as a stand-in for the iOS
static build — same `UNIRT_BUILD_SHARED=OFF` code path, different
`CMAKE_SYSTEM_NAME`.

CI also builds and runs the native static-library tests
(`unirt-static-registration-test`, `unirt-inference-smoke-test`) on the iOS
simulator on every push (see the `ios` job in `.github/workflows/ci.yml`).

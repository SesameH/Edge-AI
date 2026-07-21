# UniRT iOS binding

Swift wrapper (`UniRTKit`) over the UniRT C API: LLM (text) and VLM
(multimodal) both via llama_cpp/GGUF; embeddings follow the same pattern
when needed.

Unlike Android, there is no JNI-style glue layer: Swift calls the C ABI
directly. The one iOS-specific wrinkle is plugin loading — iOS forbids
`dlopen` of arbitrary paths, so plugins cannot be discovered from a
directory scan the way `llama_cpp` is on macOS/Linux/Windows. Instead the
app links the plugin as a **static library** and joins it in-process with
`unirt_register_plugin()` before `unirt_init()`.

## Build the XCFramework (recommended)

`bindings/ios/framework/` is a small CMake project that merges
`libunirt` + the llama_cpp plugin + llama.cpp's own `libggml*`/`libllama`/
`libmtmd` into **one dylib** per platform slice (`-force_load`ing the two
UniRT static archives so every C API entry point — and the plugin's
`unirt_plugin_id`/`unirt_plugin_open` — survive; llama.cpp's own libraries
don't need that, since the plugin genuinely calls deep into their API, so
ordinary linking already retains what's used). This is the iOS analog of
the Android binding's AAR: one artifact, nothing to copy or embed by hand.

```sh
git submodule update --init --depth 1 third-party/llama.cpp

cmake -S bindings/ios/framework -B build-ios-framework-sim -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
  -DGGML_METAL=OFF -DUNIRT_PLUGIN_MLX=OFF -DUNIRT_PLUGIN_ONNXRUNTIME=OFF
cmake --build build-ios-framework-sim -j8

cmake -S bindings/ios/framework -B build-ios-framework-device -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
  -DGGML_METAL=OFF -DUNIRT_PLUGIN_MLX=OFF -DUNIRT_PLUGIN_ONNXRUNTIME=OFF
cmake --build build-ios-framework-device -j8

xcodebuild -create-xcframework \
  -library build-ios-framework-device/libunirt_ios.dylib \
  -library build-ios-framework-sim/libunirt_ios.dylib \
  -output bindings/ios/UniRT.xcframework
```

Then add `bindings/ios` as a local Swift package dependency (Xcode: File →
Add Package Dependencies → Add Local...) and link `UniRTKit` to your app
target — that's it. `Package.swift`'s `UniRTNative` binary target picks up
`UniRT.xcframework` (built above, not checked in — same as `build-ios`) and
SPM links + embeds it automatically; no manual "Link Binary"/"Embed & Sign"
steps, no linker flags.

```swift
import UniRTKit

try UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)
try UniRT.start()
```

(Registration is still explicit — the merged dylib bundles the plugin, but
doesn't self-register at load time, matching this project's preference for
explicit calls over load-time magic elsewhere. `unirt_plugin_id`/
`unirt_plugin_open` are declared in `CUniRT`, resolved from the linked
`UniRTNative` binary target.)

## Build manually instead (no XCFramework)

If you'd rather manage the native libraries yourself, build against
`sdk/CMakeLists.txt` directly with `UNIRT_BUILD_SHARED=OFF` — everything
comes out as a plain static archive (`libunirt.a`, `libunirt_llama_cpp.a`,
and llama.cpp's own `libllama.a`/`libggml*.a`/`libmtmd.a`; none of them are
dylibs when built this way, so there's nothing to embed):

```sh
cmake -S sdk -B build-ios -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
  -DUNIRT_BUILD_SHARED=OFF -DBUILD_SHARED_LIBS=OFF \
  -DUNIRT_PYTHON_TESTS=OFF \
  -DUNIRT_PLUGIN_MLX=OFF -DUNIRT_PLUGIN_ONNXRUNTIME=OFF
cmake --build build-ios -j8
```

Add every `.a` under `build-ios/` to the app target's "Link Binary With
Libraries" build phase (static; no signing needed), add `bindings/ios` as a
local package dependency for the Swift sources only (skip the `UniRTNative`
binary target — the app is supplying the libraries itself), then register
and start as above. Both paths build from the same CMake configuration
(`UNIRT_BUILD_SHARED=OFF`); the XCFramework path just also merges everything
into one dylib instead of leaving it as several archives you link yourself.

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

`VlmSession` mirrors `LlmSession` (same actor, same registration/start
sequence) but takes multimodal turns (`ContentPart.text`/`.image`/`.audio`)
and per-request media on `VlmGenerateOptions`:

```swift
let session = try await UniRT.createVlmSession(
    modelPath: "/path/to/vision-model.gguf", mmprojPath: "/path/to/mmproj.gguf")
let prompt = try await session.applyChatTemplate([
    .user(.text("What's in this image?"), .image(path: "/path/to/photo.jpg")),
])
let reply = try await session.generate(
    prompt: prompt, options: VlmGenerateOptions(imagePaths: ["/path/to/photo.jpg"]))
```

`LlmSession`/`VlmSession` are Swift `actor`s: the native handle is
single-threaded by contract, and actor isolation confines every native call
without extra locking, mirroring the Kotlin binding's dedicated-dispatcher
approach.

Models ship however the app prefers (bundled resource, downloaded at first
run); pass an absolute filesystem path — the sandbox means that's usually
somewhere under `FileManager.default.urls(for: .documentDirectory, ...)`
or `Bundle.main`.

## Run the integration tests

`Tests/UniRTKitTests/InferenceSmokeTests.swift` is the Swift-layer
counterpart to `tests/native/test_inference_smoke.cpp`: registers the real
llama_cpp plugin, loads a GGUF model, applies the chat template, and runs
both blocking and streaming generation. `VlmLinkSmokeTests.swift` proves the
six `unirt_vlm_*` entry points actually link (no VLM test model is available
to run real multimodal inference, so it only checks that a missing model
fails cleanly through the whole chain rather than link-erroring or
crashing).

Once `UniRT.xcframework` is built (see above), both just run:

```sh
export TEST_RUNNER_UNIRT_TEST_MODEL_PATH="/absolute/path/to/SmolLM2-135M-Instruct-Q8_0.gguf"
xcodebuild test -scheme UniRTKit -destination "id=$SIM_UDID"   # or 'platform=macOS' to
                                                                 # sanity-check without a simulator
```

(`TEST_RUNNER_`-prefixed variables are xcodebuild's mechanism for passing
environment into the test process; `InferenceSmokeTests` `XCTSkip`s without
one.) No linker flags needed — that's the point of the binary target.

This exact mechanism (XCFramework + SPM binary target linking with zero
manual flags) was verified end to end against a real model, using a
`macos-arm64` XCFramework slice as a stand-in destination (`platform=macOS`)
since this dev machine had no iOS Simulator runtime installed — the real
`ios-arm64`/`ios-arm64-simulator` slices were verified separately for
correct static merging (`otool -L` shows no llama.cpp dylib dependencies,
only system frameworks) and symbol presence (`nm -gU` for every
`unirt_llm_*`/`unirt_vlm_*`/`unirt_plugin_*` entry point), but not run
end-to-end on a real Simulator by this session.

CI also builds and runs the native static-library tests
(`unirt-static-registration-test`, `unirt-inference-smoke-test`) on the iOS
simulator on every push (see the `ios` job in `.github/workflows/ci.yml`).

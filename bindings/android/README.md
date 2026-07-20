# UniRT Android binding

Kotlin + JNI layer over the UniRT C API. Text generation (llama_cpp / GGUF)
only for now; VLM and embeddings follow the same pattern when needed.

## Build the native libraries

```sh
cmake -S bindings/android/jni -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUNIRT_PLUGIN_MLX=OFF -DUNIRT_PLUGIN_ONNXRUNTIME=OFF
cmake --build build-android -j8
```

Copy into your app module:

- `build-android/sdk/src/libunirt.so` → `src/main/jniLibs/arm64-v8a/`
- `build-android/plugins/libunirt_plugin_llama_cpp.so` → same dir
- `build-android/libunirt_jni.so` → same dir
- llama.cpp's `libllama.so` / `libggml*.so` from the build tree → same dir
- `bindings/android/kotlin/ai/unirt/**` → your source set

Plugins are discovered automatically: the registry scans the directory that
holds `libunirt.so` (the app's native lib dir) for the flat
`libunirt_plugin_<id>.so` naming — no environment variables needed.

## Use

Requires `kotlinx-coroutines-core`. `LlmSession` is an interface (fake it in
unit tests — local JVM tests cannot load the native library); the bundled
implementation confines all native calls to one thread per session, so every
member is safe to call from any coroutine.

```kotlin
UniRT.start()
UniRT.createLlmSession("/data/local/tmp/SmolLM2-135M-Instruct-Q8_0.gguf").use { session ->
    val prompt = session.applyChatTemplate(
        listOf(ChatMessage.user("What is the capital of France?"))
    )
    session.stream(prompt).collect { piece -> print(piece) }   // cancel to stop decoding
}
UniRT.stop()
```

Kotlin conventions over ceremony: default arguments instead of builders,
`object` instead of a singleton class, `Flow` instead of listener
interfaces.

Models ship however the app prefers (assets, download at first run); pass an
absolute filesystem path. CI cross-compiles this binding for arm64-v8a on
every push; on-device instrumentation tests are future work.

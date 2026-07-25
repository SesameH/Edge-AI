# UniRT Android binding

Kotlin + JNI layer over the UniRT C API. LLM (text) and VLM (multimodal)
both via llama_cpp/GGUF; embeddings follow the same pattern when needed.

## Build the AAR

`bindings/android` is a real Android library module (Gradle + AGP,
`com.android.library`) — `externalNativeBuild` drives `jni/CMakeLists.txt`
itself, so the resulting AAR already bundles `libunirt.so`,
`libunirt_plugin_llama_cpp.so`, `libunirt_jni.so`, llama.cpp's own
`libggml*`/`libllama`/`libmtmd`, and `libomp.so`, all under `jni/arm64-v8a/`.

```sh
cd bindings/android
./gradlew assembleRelease   # -> build/outputs/aar/unirt-android-release.aar
./gradlew test              # unit tests: fakes LlmSession/VlmSession — anything
                             # touching Native itself needs a real device/emulator
```

Needs `ANDROID_HOME`/`ANDROID_SDK_ROOT` set (NDK 27.0.12077973, matched in
`build.gradle.kts`'s `ndkVersion`) and a JDK 17. Add the AAR as a dependency
in your app module — plugins are still discovered automatically at runtime
(the registry scans the directory holding `libunirt.so` for the flat
`libunirt_plugin_<id>.so` naming), no environment variables needed.

Only `arm64-v8a` is built today (`abiFilters` in `build.gradle.kts`), matching
what CI actually tests; add more ABIs there if you need them, untested.

## Build manually instead (no Gradle/AAR)

If you'd rather manage the native libraries yourself:

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

Both paths build from the same `jni/CMakeLists.txt` — the AAR path is just
AGP driving it instead of you doing so by hand.

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
    session.stream(prompt).collect { event ->
        when (event) {
            is LlmStreamResult.Token -> print(event.text)                  // cancel to stop decoding
            is LlmStreamResult.Completed -> println("\n${event.profile}")  // ttft, tok/s, stop reason
            is LlmStreamResult.Error -> println("\ngeneration failed: ${event.cause}")
        }
    }
}
UniRT.stop()
```

`VlmSession` mirrors `LlmSession` — same threading contract, same
`LlmStreamResult` stream events — but takes multimodal turns (`ContentPart.Text`/
`Image`/`Audio`) and per-request media on `VlmGenerateOptions` instead of
`GenerateOptions` (kept separate: image/audio fields would be dead weight on
every LLM call):

```kotlin
UniRT.start()
UniRT.createVlmSession(
    modelPath = "/data/local/tmp/vision-model.gguf",
    mmprojPath = "/data/local/tmp/mmproj.gguf",
).use { session ->
    val prompt = session.applyChatTemplate(
        listOf(VlmChatMessage.user(
            ContentPart.Text("What's in this image?"),
            ContentPart.Image("/data/local/tmp/photo.jpg"),
        ))
    )
    val reply = session.generate(
        prompt,
        VlmGenerateOptions(imagePaths = listOf("/data/local/tmp/photo.jpg")),
    )
    println(reply)
}
UniRT.stop()
```

### Tool calling

The declared tools are compiled into a JSON schema the sampler physically
cannot leave, so a reply always parses and always names a tool you declared —
the same approach the Python binding and the OpenAI-compatible server take,
and all three emit an identical schema for identical tools. One call per turn.

```kotlin
val tools = listOf(
    ToolDefinition(
        name = "get_weather",
        description = "Look up the weather",
        parametersJson = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""",
    )
)
var messages = listOf(ChatMessage.user("What's the weather in Taipei?"))

when (val reply = session.chatWithTools(messages, tools)) {
    is ToolReply.Text -> println(reply.content)
    is ToolReply.Call -> {
        val result = runTool(reply.call.name, reply.call.argumentsJson)   // your code
        messages = messages + ChatMessage.toolCall(reply.call) +
            ChatMessage.toolResult(reply.call.name, result)
        println(session.chatWithTools(messages, tools))   // the model sees the result
    }
}
```

`parametersJson` is JSON Schema text embedded verbatim, so property order
survives exactly as written; `null` means the tool takes no arguments.
`ToolChoice.Required` forces some call, `ToolChoice.Function(name)` forces one
specific tool, and `ToolChoice.None` runs the turn as plain chat. Tools spend
the same slot as `grammar`/`jsonMode`/`jsonSchema`, so setting both throws.

Kotlin conventions over ceremony: default arguments instead of builders,
`object` instead of a singleton class, `Flow` instead of listener
interfaces.

Models ship however the app prefers (assets, download at first run); pass an
absolute filesystem path. CI builds the AAR and runs its unit tests on every
push, alongside a standalone native NDK build exercising the manual path
above; on-device instrumentation tests are future work.

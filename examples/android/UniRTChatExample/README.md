# UniRT Chat (Android example)

A Jetpack Compose app that consumes the Android binding as a prebuilt AAR
and runs a streaming chat loop entirely on-device — one app, a **Text /
Vision** switch at the top toggles between an `LlmSession` (SmolLM2) and a
`VlmSession` (LFM2-VL-450M) loaded from APK assets. Same layout and visual
language as the iOS example (`examples/ios/UniRTChatExample`), so the two
demos read as one product.

Verified end to end on the Android emulator (arm64-v8a system image on
Apple Silicon — runs natively, no translation): model load, greedy
generation, a real reply mentioning "Paris" in Text mode, and a real
description of the bundled `test-photo.jpg` in Vision mode, via the
instrumentation test below. Emulator numbers are functional-only; treat a
real device as the source of truth for performance.

![Android emulator running both modes: Text asking the capital of France, then switching to Vision to describe the bundled test image — driven by the instrumentation test while `adb shell screenrecord` captured the screen](docs/emulator-demo.gif)

## Build & run

```sh
# 1. Build the binding AAR the app depends on (once, or after native changes)
cd bindings/android && ./gradlew assembleRelease && cd -

# 2. Bundle test models as assets (any GGUF works; these are small defaults)
cd examples/android/UniRTChatExample/app/src/main/assets
curl -L -o model.gguf \
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
curl -L -o vlm-model.gguf \
  "https://huggingface.co/runanywhere/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-Q4_0.gguf"
curl -L -o mmproj.gguf \
  "https://huggingface.co/runanywhere/LFM2-VL-450M-GGUF/resolve/main/mmproj-LFM2-VL-450M-Q8_0.gguf"
cd -
# test-photo.jpg is already checked in (same synthetic scene as the iOS example)

# 3. Build, install, launch (any connected device or running emulator)
cd examples/android/UniRTChatExample
./gradlew installDebug
adb shell am start -n ai.unirt.example.chat/.MainActivity
```

Needs `ANDROID_HOME`/a `local.properties` with `sdk.dir`, and a JDK 17–21
(`JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"`
works if Android Studio is installed).

No real device? An arm64 emulator is enough for everything functional:

```sh
~/Library/Android/sdk/emulator/emulator -avd <name> &   # AVD with an arm64-v8a image
adb wait-for-device
```

## UI test

`ChatUiTest` mirrors the iOS example's `ChatUITests`: one continuous
session — types the capital-of-France question in Text mode and waits for
"Paris", then switches to Vision, attaches the test photo, and waits for a
reply describing the scene. Real inference, not a render smoke test:

```sh
./gradlew connectedDebugAndroidTest
```

## Record a demo GIF

`docs/emulator-demo.gif` was made by letting the UI test drive the app
while the emulator recorded its own screen — fully scripted, no hand on
the wheel:

```sh
adb shell "screenrecord --time-limit 170 /sdcard/demo.mp4" &
./gradlew connectedDebugAndroidTest --rerun-tasks
adb shell pkill -INT screenrecord; adb pull /sdcard/demo.mp4
ffmpeg -ss <start> -to <end> -i demo.mp4 \
  -vf "fps=10,scale=360:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  docs/emulator-demo.gif
```

## How it works

`ChatViewModel` mirrors the iOS example: two sessions (`LlmSession` +
`VlmSession`), each opened lazily the first time its mode is selected and
kept resident for the app's lifetime — switching modes never re-pays model
load or its memory spike. Streaming goes through the binding's
`Flow<LlmStreamResult>`; the stats row shows the `GenerationProfile` from
the last completed generation (TTFT, prefill/decode speed, token counts,
stop reason).

Models ship in APK assets, but the native layer wants filesystem paths, so
each asset is copied to `filesDir` once on first use. Two packaging choices
matter (`app/build.gradle.kts`):

- `noCompress += "gguf"` — weights are already packed; APK compression
  would only burn build time.
- `jniLibs.useLegacyPackaging = true` — plugin discovery scans the
  directory holding `libunirt.so` for `libunirt_plugin_*.so`; the modern
  default keeps `.so` files inside `base.apk` (no directory to scan) and
  `UniRT.start()` fails with "filesystem error: in canonical".

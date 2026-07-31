# Releasing without CI

`.github/workflows/publish-sdk.yml` builds every platform, syncs wrapper source
into SesameH/unirt-sdk, attaches the release assets and publishes to PyPI. When
it cannot run, this directory is how the same release gets made by hand. The
workflow stays the source of truth; these scripts mirror individual steps of it.

v0.2.2 was released this way. What follows is that sequence, in order.

## 0. Work directory

Per-release scratch lives in `.release-work-v<version>/` at the repo root
(gitignored). Nothing here writes into it automatically -- it is just where the
wheels and assets are collected.

## 1. Build the wheels

**Every `cmake -S sdk` for a release must pass `-DUNIRT_VERSION=v<version>.**
It defaults to `v0.0.0`, and that default is what `unirt version` reports back
to the user as the SDK version — 0.2.2 shipped saying `SDK: v0.0.0`.

**A release build must also produce every runtime the docs promise.** Both
optional plugins are skipped silently when their dependency is missing —
`UNIRT_PLUGIN_MLX` needs a separately built static MLX, and the ONNX Runtime
plugin prints a warning and `return()`s when `UNIRT_ONNXRUNTIME_ROOT` is unset.
A wheel with a runtime quietly missing looks identical to one without, so check
the install prefix before building the wheel:

```sh
cmake -S third-party/mlx -B build-mlx -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX="$PWD/build-mlx/install"
cmake --build build-mlx -j8 && cmake --install build-mlx      # macOS only

ORT=1.26.0                                    # per-platform SDK, see README.md
cmake -S sdk -B build -DCMAKE_BUILD_TYPE=Release -DUNIRT_VERSION=v<version> \
    -DUNIRT_ONNXRUNTIME_ROOT="$PWD/build-onnxruntime/onnxruntime-osx-arm64-$ORT"
cmake --build build -j8 && cmake --install build --prefix sdk/pkg-unirt

ls sdk/pkg-unirt/lib/*/            # llama_cpp, mlx (macOS), onnxruntime
```

The published wheels carry `llama_cpp` and `onnxruntime` everywhere, plus `mlx`
on macOS. `unirt devices` from the installed wheel is the end-to-end check.

Per-platform build commands are in `publish-sdk.yml`; the common entry point is:

```sh
python3 bindings/python/build_wheel.py \
    --native-prefix <prefix with lib/> \
    --version <X.Y.Z> \
    --platform-tag <tag> \
    --output-dir <out>
```

The native libraries come from a `cmake --install` prefix built for that
platform. If nothing under `sdk/`, `plugin/` or `third-party/` changed since the
previous tag, the previous release's natives can be reused -- prove it with an
empty diff before claiming it in the notes:

```sh
git diff v<prev>..HEAD -- sdk/ plugin/ third-party/ ':(exclude)sdk/plugins/mlx'
```

`build_wheel.py` derives native library names from the *target* platform tag,
not the build host, so the Windows wheels can be assembled on macOS.

## 2. Verify every wheel

A wheel that has not run inference has not been verified. `twine check` and a
successful `pip install` prove neither.

| Platform | How |
|---|---|
| macOS ARM64 | run `bindings/python/smoke_inference_wheel.py` on the host |
| Linux ARM64 / x86_64 | Debian 11 container (glibc 2.31 -- the manylinux tag's actual floor, not Ubuntu 22.04's 2.35); x86_64 runs under qemu-user |
| Windows ARM64 / x86_64 | one Windows 11 ARM64 VM, see [windows-verify/](windows-verify/) |

Two traps, both hit for real during v0.2.2:

- **Always pass `--platform` to `docker run`.** Docker will happily serve a
  cached `linux/amd64` image for an `arm64` request, and the wheel then fails
  for the wrong reason -- or worse, passes.
- **Colima mounts only `$HOME`.** A container path under `/private/tmp` is
  silently empty, and `compileall` on an empty directory exits 0. Keep release
  artifacts under `$HOME`.

Also run `auditwheel show` on the Linux wheels and check the reported tag
matches the filename.

## 3. macOS native tarball

`unirt-python-native-macos-arm64.tar.gz` is the `lib/` tree the Python binding
expects, at the archive root:

```sh
tar -C sdk/pkg-unirt -czf unirt-python-native-macos-arm64.tar.gz lib
```

It must be the same binaries the macOS wheel ships. Check it rather than
assuming -- if the wheel reused a previous release's natives and the tarball was
rebuilt (or vice versa), they will not match:

```sh
mkdir -p /tmp/cmp/tar /tmp/cmp/whl
tar -C /tmp/cmp/tar -xzf unirt-python-native-macos-arm64.tar.gz
(cd /tmp/cmp/whl && unzip -q unirt-<version>-py3-none-macosx_14_0_arm64.whl 'unirt/lib/*')
diff <(cd /tmp/cmp/tar/lib && find . -type f | sort | xargs shasum -a 256) \
     <(cd /tmp/cmp/whl/unirt/lib && find . -type f | sort | xargs shasum -a 256)
```

## 4. iOS XCFramework

Two builds, one per sysroot, then combined:

```sh
for target in "iphonesimulator sim" "iphoneos device"; do
    set -- $target
    cmake -S bindings/ios/framework -B build-ios-framework-$2 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=$1 \
        -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
        -DGGML_METAL=OFF -DUNIRT_PLUGIN_MLX=OFF -DUNIRT_PLUGIN_ONNXRUNTIME=OFF
    cmake --build build-ios-framework-$2 -j8
done
xcodebuild -create-xcframework \
    -library build-ios-framework-device/libunirt_ios.dylib \
    -library build-ios-framework-sim/libunirt_ios.dylib \
    -output publish/UniRT.xcframework
(cd publish && zip -q -r -y unirt-ios-xcframework.zip UniRT.xcframework)
```

`plutil -p publish/UniRT.xcframework/Info.plist` should list both `ios-arm64`
and `ios-arm64-simulator`.

## 5. Android AAR (two stages)

The published AAR is **not** the one this repo builds. This repo's AAR supplies
the prebuilt natives; unirt-sdk rebuilds `libunirt_jni.so` from its own synced
source and packages the result. Doing only the first stage ships a JNI library
built from sources the consumer cannot see.

```sh
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"  # JDK 21; AGP rejects 23
export ANDROID_HOME="$HOME/Library/Android/sdk"

(cd bindings/android && ./gradlew assembleRelease --console=plain)

mkdir -p publish/arm64-v8a
unzip -o -j "$(find bindings/android/build/outputs/aar -name '*.aar')" \
    'jni/arm64-v8a/*' -d publish/arm64-v8a
rm -f publish/arm64-v8a/libunirt_jni.so     # unirt-sdk rebuilds this one
tar -C publish -czf publish/unirt-android-native-arm64-v8a.tar.gz arm64-v8a
```

Then, after step 6, in the unirt-sdk clone:

```sh
mkdir -p android/prebuilt/arm64-v8a
tar -C android/prebuilt/arm64-v8a -xzf <core>/publish/unirt-android-native-arm64-v8a.tar.gz \
    --strip-components=1
(cd android && ./gradlew assembleRelease --console=plain)
```

Sanity-check the result contains what the release claims, e.g. for a release
adding tool calling:

```sh
unzip -p android/build/outputs/aar/*.aar classes.jar > /tmp/c.jar && unzip -l /tmp/c.jar | grep Tool
```

## 6. Sync wrapper source

```sh
git clone https://github.com/SesameH/unirt-sdk.git
./tools/release/sync-wrapper-source.sh ./unirt-sdk
cd unirt-sdk && git status        # review, then commit and push
```

CI commits this as `sync wrapper source from core v<version>`.

## 7. Tag and release

```sh
git tag v<version> <commit> && git push origin v<version>
gh release create v<version> --repo SesameH/unirt-sdk --title v<version> \
    --notes-file NOTES.md \
    unirt-python-native-macos-arm64.tar.gz unirt-android.aar \
    unirt-ios-xcframework.zip ./*.whl
```

Eight assets: five wheels, the AAR, the XCFramework, the macOS native tarball.
Compare against the previous release before publishing -- a missing asset reads
as a regression to anyone consuming it.

## 8. PyPI

```sh
python3 -m twine check ./*.whl
python3 -m twine upload ./*.whl        # credentials in ~/.pypirc, chmod 600
```

A PyPI API token is ~200 characters and starts with `pypi-`. Anything shorter is
not a token and produces a bare `403 Forbidden`.

**A version number on PyPI is permanent.** It cannot be overwritten or re-used
after deletion; the only remedy for a bad release is to yank it, which the web
UI does and the API does not:
<https://pypi.org/manage/project/unirt/release/X.Y.Z/> -> Options -> Yank.

The simple index lags the upload by a few seconds, so an immediate
`pip install unirt` can still resolve the previous version. Re-check rather than
assuming the upload was wrong.

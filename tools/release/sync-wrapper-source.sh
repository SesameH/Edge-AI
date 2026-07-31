#!/bin/sh
# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copy the wrapper source this repo publishes into a clone of SesameH/unirt-sdk.
# Nothing under sdk/, sdk/plugins/ or third-party/ is ever copied -- unirt-sdk
# only ever sees bindings, the frozen header, and the examples.
#
# This mirrors the "Sync wrapper source" step of .github/workflows/publish-sdk.yml.
# Keep the two in step: the workflow is the source of truth when CI can run,
# this script is what you use when it cannot.
#
# Usage:  ./sync-wrapper-source.sh <path-to-unirt-sdk-clone>
#
# Review `git status` in the clone afterwards, then commit and push yourself.

set -eu

if [ $# -ne 1 ]; then
    echo "usage: $0 <path-to-unirt-sdk-clone>" >&2
    exit 2
fi

sdk=$(cd "$1" && pwd)
core=$(cd "$(dirname "$0")/../.." && pwd)

[ -d "$sdk/.git" ] || { echo "$sdk is not a git clone" >&2; exit 1; }
{ [ -d "$sdk/python" ] && [ -d "$sdk/android" ]; } || {
    echo "$sdk does not look like unirt-sdk (no python/ or android/)" >&2
    exit 1
}

# BSD sed needs an explicit (empty) backup suffix for -i; GNU sed, as used in
# CI, must not have one. That is the only difference from the workflow.
sedi() { sed -i '' "$@"; }

rsync -a --delete --exclude unirt/lib --exclude LICENSE \
    "$core/bindings/python/" "$sdk/python/"
rsync -a --delete "$core/bindings/android/kotlin/" "$sdk/android/kotlin/"
rsync -a --delete "$core/bindings/android/test/" "$sdk/android/test/"
cp "$core/bindings/android/jni/unirt_jni.cpp" "$sdk/android/jni/unirt_jni.cpp"
rsync -a --delete "$core/bindings/ios/Sources/UniRTKit/" "$sdk/ios/Sources/UniRTKit/"
rsync -a --delete "$core/bindings/ios/Tests/" "$sdk/ios/Tests/"
cp "$core/sdk/include/unirt.h" "$sdk/include/unirt.h"
cp "$core/sdk/include/unirt.h" "$sdk/ios/Sources/CUniRT/unirt.h"

mkdir -p "$sdk/examples"
cp "$core/examples/chat.html" "$sdk/examples/chat.html"
rsync -a --delete \
    --exclude '*.gguf' --exclude '*.xcodeproj' \
    --exclude 'UniRTChatExample/Info.plist' --exclude '.swiftpm' \
    "$core/examples/ios/UniRTChatExample/" "$sdk/examples/ios/UniRTChatExample/"
# unirt-sdk's Swift package lives at ios/, not bindings/ios/
sedi 's#path: \.\./\.\./\.\./bindings/ios#path: ../../../ios#' \
    "$sdk/examples/ios/UniRTChatExample/project.yml"

rsync -a --delete \
    --exclude '*.gguf' --exclude 'build' --exclude '.gradle' \
    --exclude '.cxx' --exclude '.kotlin' --exclude 'local.properties' \
    "$core/examples/android/UniRTChatExample/" "$sdk/examples/android/UniRTChatExample/"
# unirt-sdk's Android library module lives at android/, not bindings/android/
sedi 's#bindings/android/build/outputs#android/build/outputs#' \
    "$sdk/examples/android/UniRTChatExample/app/build.gradle.kts"
sedi 's|cd bindings/android && ./gradlew assembleRelease && cd -|cd android \&\& ./gradlew assembleRelease \&\& cd -   # needs prebuilt/<abi>/*.so, see android/README.md|' \
    "$sdk/examples/android/UniRTChatExample/README.md"

echo "synced into $sdk -- review 'git status' there before committing"

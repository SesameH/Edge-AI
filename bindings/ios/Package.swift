// swift-tools-version:5.9
// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import PackageDescription

let package = Package(
    name: "UniRTKit",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "UniRTKit", targets: ["UniRTKit"]),
    ],
    targets: [
        // Exposes sdk/include/unirt.h as a Clang module. No sources of its
        // own; the app target supplies the static libraries built by
        // `cmake -S sdk -B build-ios -DCMAKE_SYSTEM_NAME=iOS ...` (see
        // README.md) at the Xcode project level.
        .systemLibrary(name: "CUniRT"),
        // Built by `cmake --build bindings/ios/framework` (device + simulator)
        // then `xcodebuild -create-xcframework` — see README.md's "Build the
        // XCFramework" section. Not checked in (like build-ios, it's a build
        // artifact); build it before `swift build`/`xcodebuild` resolves this.
        .binaryTarget(name: "UniRTNative", path: "UniRT.xcframework"),
        .target(name: "UniRTKit", dependencies: ["CUniRT", "UniRTNative"]),
        // Runs one generation against a GGUF model — see README.md's "Run
        // the integration test" section for the required environment.
        .testTarget(name: "UniRTKitTests", dependencies: ["UniRTKit", "CUniRT"]),
    ]
)

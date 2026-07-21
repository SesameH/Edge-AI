// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// Real Android library module — ./gradlew assembleRelease produces an AAR
// with the JNI glue + llama_cpp plugin + libunirt (and llama.cpp's own
// libggml*/libllama/libmtmd) already bundled per ABI, via externalNativeBuild
// pointing at the same jni/CMakeLists.txt the manual "copy .so files in"
// workflow in README.md uses — both paths build from the identical CMake
// project, just driven differently (AGP drives this one; the README's manual
// path is for apps that would rather manage native libs themselves).
//
// LlmSession/VlmSession are interfaces specifically so they're fakeable in
// test/: Native's System.loadLibrary("unirt_jni") means anything touching
// NativeLlmSession/NativeVlmSession/Native directly cannot run in a local
// unit test (no device/emulator) — those need instrumentation tests (future
// work). Local unit tests need no android.* framework classes here (the
// sources have none), so they run as plain JVM tests without Robolectric.

plugins {
    id("com.android.library") version "8.7.3"
    kotlin("android") version "2.0.21"
}

android {
    namespace = "ai.unirt"
    compileSdk = 35
    ndkVersion = "27.0.12077973"

    defaultConfig {
        minSdk = 28 // matches ANDROID_PLATFORM=android-28 used everywhere else in this repo

        externalNativeBuild {
            cmake {
                abiFilters += "arm64-v8a"
                arguments += listOf(
                    "-DUNIRT_PLUGIN_MLX=OFF",
                    "-DUNIRT_PLUGIN_ONNXRUNTIME=OFF",
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("jni/CMakeLists.txt")
        }
    }

    sourceSets {
        getByName("main") {
            kotlin.srcDirs("kotlin")
        }
        getByName("test") {
            kotlin.srcDirs("test")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    @Suppress("UnstableApiUsage")
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
    testImplementation(kotlin("test"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
}

tasks.withType<Test> {
    useJUnitPlatform()
}

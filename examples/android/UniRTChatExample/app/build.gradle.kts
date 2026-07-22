// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// Consumes the binding as a prebuilt AAR (build it first:
// `cd bindings/android && ./gradlew assembleRelease`) rather than a
// composite build — the example then exercises exactly what an external
// app would depend on, native libs and all.

plugins {
    id("com.android.application") version "8.7.3"
    kotlin("android") version "2.0.21"
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21"
}

android {
    namespace = "ai.unirt.example.chat"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.unirt.example.chat"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += "arm64-v8a" }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // GGUF weights are already quantized/packed; recompressing them into the
    // APK wastes minutes of build time and breaks mmap-style streaming reads.
    androidResources { noCompress += "gguf" }

    // Plugin discovery scans the directory holding libunirt.so for
    // libunirt_plugin_*.so; with the modern packaging default the libs stay
    // inside base.apk (dlopen path "base.apk!/lib/...") and there is no
    // directory to scan, so runtime.init fails with "filesystem error: in
    // canonical". Extracting them to disk restores a real plugin root.
    packaging { jniLibs { useLegacyPackaging = true } }

    buildFeatures { compose = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation(files("../../../../bindings/android/build/outputs/aar/unirt-android-release.aar"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")

    androidTestImplementation(platform("androidx.compose:compose-bom:2024.12.01"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}

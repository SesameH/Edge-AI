// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// Plain Kotlin/JVM module, not an Android library module: bindings/android/kotlin
// has zero android.* imports (System.loadLibrary is plain java.lang), so there's
// nothing here that needs the Android Gradle Plugin or an Android SDK. This exists
// so the Kotlin sources are actually compiled and unit-tested by something (CI's
// `android` job only ever NDK-builds the native side) — an app consuming this
// binding still copies kotlin/ai/unirt/** into its own Android module per README.md;
// this project is for verifying the sources, not for shipping an AAR.
//
// LlmSession/VlmSession are interfaces specifically so they're fakeable here:
// Native's System.loadLibrary("unirt_jni") means anything touching NativeLlmSession/
// NativeVlmSession/Native directly cannot run in a plain JVM test — those need a
// real device/emulator (future work, same as Android instrumentation tests).

plugins {
    kotlin("jvm") version "2.0.21"
}

repositories {
    mavenCentral()
}

kotlin {
    // Cross-target bytecode level only (no Gradle toolchain provisioning):
    // whatever JDK runs Gradle can emit Java 17 class files directly, so
    // this doesn't require a JDK 17 install on top of it.
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
    sourceSets {
        main {
            kotlin.srcDirs("kotlin")
        }
        test {
            kotlin.srcDirs("test")
        }
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
    testImplementation(kotlin("test"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
}

tasks.test {
    useJUnitPlatform()
}

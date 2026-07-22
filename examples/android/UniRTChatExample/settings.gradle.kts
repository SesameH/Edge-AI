// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "UniRTChatExample"
include(":app")

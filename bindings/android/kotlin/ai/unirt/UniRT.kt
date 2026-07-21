// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

import ai.unirt.internal.NativeLlmSession
import ai.unirt.internal.NativeVlmSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Runtime lifecycle and session factory.
 *
 * ```kotlin
 * UniRT.start()
 * UniRT.createLlmSession("/data/.../model.gguf").use { session ->
 *     session.stream(session.applyChatTemplate(listOf(ChatMessage.user("Hi"))))
 *         .collect { piece -> print(piece) }
 * }
 * UniRT.stop()
 * ```
 */
object UniRT {
    /** Scan and load the bundled backend plugins. Call once, off the main
     *  thread or from any coroutine. */
    suspend fun start() = withContext(Dispatchers.IO) { check(Native.init()) }

    /** Unload plugins; every session must be closed first. */
    suspend fun stop() = withContext(Dispatchers.IO) { check(Native.deinit()) }

    fun version(): String = Native.version()

    fun plugins(): List<String> = Native.pluginList().toList()

    /** Load a model and return its session. Heavy: minutes-old phones and
     *  multi-GB models mean seconds of IO + prefill, hence suspend. */
    suspend fun createLlmSession(
        modelPath: String,
        pluginId: String = "llama_cpp",
        deviceId: String? = null,
        nCtx: Int = 0,
        nGpuLayers: Int = -1,
    ): LlmSession = NativeLlmSession.open(modelPath, pluginId, deviceId, nCtx, nGpuLayers)

    /** Load a multimodal model and return its session. */
    suspend fun createVlmSession(
        modelPath: String,
        mmprojPath: String? = null,
        pluginId: String = "llama_cpp",
        deviceId: String? = null,
        nCtx: Int = 0,
        nGpuLayers: Int = -1,
    ): VlmSession = NativeVlmSession.open(modelPath, mmprojPath, pluginId, deviceId, nCtx, nGpuLayers)

    private fun check(code: Int) {
        if (code < 0) {
            throw UniRTException(code, Native.errorMessage(code) + ' ' + Native.lastError())
        }
    }
}

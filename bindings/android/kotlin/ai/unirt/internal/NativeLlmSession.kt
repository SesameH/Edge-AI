// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt.internal

import ai.unirt.ChatMessage
import ai.unirt.GenerateOptions
import ai.unirt.LlmSession
import ai.unirt.Native
import ai.unirt.TokenCallback
import ai.unirt.UniRTException
import java.util.concurrent.Executors
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

/**
 * The one [LlmSession] implementation, over the JNI surface. The native
 * handle is single-threaded by contract, so every native call is funnelled
 * through [dispatcher]; suspending members are therefore callable from any
 * coroutine without external locking.
 */
internal class NativeLlmSession private constructor(
    private var handle: Long,
    private val dispatcher: ExecutorCoroutineDispatcher,
) : LlmSession {

    companion object {
        suspend fun open(
            modelPath: String,
            pluginId: String,
            deviceId: String?,
            nCtx: Int,
            nGpuLayers: Int,
        ): NativeLlmSession {
            val dispatcher = Executors.newSingleThreadExecutor { runnable ->
                Thread(runnable, "unirt-llm").apply { isDaemon = true }
            }.asCoroutineDispatcher()
            val handle = withContext(dispatcher) {
                Native.llmCreate(modelPath, pluginId, deviceId, nCtx, nGpuLayers)
            }
            if (handle == 0L) {
                dispatcher.close()
                throw UniRTException(-1, "cannot load $modelPath: ${Native.lastError()}")
            }
            return NativeLlmSession(handle, dispatcher)
        }
    }

    private fun requireOpen(): Long {
        check(handle != 0L) { "session is closed" }
        return handle
    }

    private fun raise(): Nothing = throw UniRTException(-1, Native.lastError())

    override suspend fun applyChatTemplate(
        messages: List<ChatMessage>,
        addGenerationPrompt: Boolean,
    ): String = withContext(dispatcher) {
        Native.llmApplyChatTemplate(
            requireOpen(),
            messages.map { it.role }.toTypedArray(),
            messages.map { it.content }.toTypedArray(),
            addGenerationPrompt,
        ) ?: raise()
    }

    override suspend fun generate(prompt: String, options: GenerateOptions): String =
        withContext(dispatcher) {
            Native.llmGenerate(
                requireOpen(), prompt, options.maxTokens, options.temperature,
                options.topP, options.topK, options.seed, null,
            ) ?: raise()
        }

    override fun stream(prompt: String, options: GenerateOptions): Flow<String> =
        channelFlow {
            withContext(dispatcher) {
                Native.llmGenerate(
                    requireOpen(), prompt, options.maxTokens, options.temperature,
                    options.topP, options.topK, options.seed,
                    TokenCallback { piece -> trySendBlocking(piece).isSuccess },
                ) ?: raise()
            }
        }

    override suspend fun reset() {
        val status = withContext(dispatcher) { Native.llmReset(requireOpen()) }
        if (status < 0) throw UniRTException(status, Native.errorMessage(status))
    }

    override fun close() {
        if (handle == 0L) return
        runBlocking(dispatcher) {
            Native.llmDestroy(handle)
            handle = 0L
        }
        dispatcher.close()
    }
}

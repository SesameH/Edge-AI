// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt.internal

import ai.unirt.ChatMessage
import ai.unirt.GenerateOptions
import ai.unirt.LlmGenerateResult
import ai.unirt.LlmSession
import ai.unirt.LlmStreamResult
import ai.unirt.Native
import ai.unirt.RuntimeStats
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

    /** Unpacks [GenerateOptions] into the native call once, instead of at
     *  every call site — [Native.llmGenerate] itself stays flat/scalar since
     *  that's what a JNI signature should be (an object parameter there would
     *  mean unpacking it in C++ via GetFieldID by string name: no compile-time
     *  check, easy to typo, no real win over the JVM marshalling primitives
     *  directly). */
    private fun nativeGenerate(
        prompt: String,
        options: GenerateOptions,
        onToken: TokenCallback?,
    ): LlmGenerateResult = Native.llmGenerate(
        requireOpen(), prompt, options.maxTokens, options.temperature,
        options.topP, options.topK, options.seed, options.grammar,
        options.jsonMode, options.jsonSchema, onToken,
    ) ?: raise()

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
        withContext(dispatcher) { nativeGenerate(prompt, options, onToken = null).text }

    // The C ABI has one callback (per token) and signals completion only by
    // returning from the blocking call — there is no separate "done" hook.
    // So Completed/Error are emitted here, after llmGenerate() itself returns,
    // using its actual result — not from inside the token callback.
    override fun stream(prompt: String, options: GenerateOptions): Flow<LlmStreamResult> =
        channelFlow {
            withContext(dispatcher) {
                try {
                    val onToken = TokenCallback { piece ->
                        trySendBlocking(LlmStreamResult.Token(piece)).isSuccess
                    }
                    val result = nativeGenerate(prompt, options, onToken)
                    trySendBlocking(LlmStreamResult.Completed(result.profile))
                } catch (e: UniRTException) {
                    trySendBlocking(LlmStreamResult.Error(e))
                }
            }
        }

    override suspend fun reset() {
        val status = withContext(dispatcher) { Native.llmReset(requireOpen()) }
        if (status < 0) throw UniRTException(status, Native.errorMessage(status))
    }

    override suspend fun runtimeStats(): RuntimeStats =
        withContext(dispatcher) { Native.llmRuntimeStats(requireOpen()) ?: raise() }

    override fun close() {
        if (handle == 0L) return
        runBlocking(dispatcher) {
            Native.llmDestroy(handle)
            handle = 0L
        }
        dispatcher.close()
    }
}

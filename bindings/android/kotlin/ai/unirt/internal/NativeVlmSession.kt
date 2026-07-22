// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt.internal

import ai.unirt.LlmGenerateResult
import ai.unirt.LlmStreamResult
import ai.unirt.Native
import ai.unirt.RuntimeStats
import ai.unirt.TokenCallback
import ai.unirt.UniRTException
import ai.unirt.VlmCapabilities
import ai.unirt.VlmChatMessage
import ai.unirt.VlmGenerateOptions
import ai.unirt.VlmSession
import ai.unirt.wireText
import ai.unirt.wireType
import java.util.concurrent.Executors
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

/**
 * The one [VlmSession] implementation, over the JNI surface. Same shape as
 * [NativeLlmSession]: the native handle is single-threaded by contract, so
 * every native call is funnelled through [dispatcher].
 */
internal class NativeVlmSession private constructor(
    private var handle: Long,
    private val dispatcher: ExecutorCoroutineDispatcher,
) : VlmSession {

    companion object {
        suspend fun open(
            modelPath: String,
            mmprojPath: String?,
            pluginId: String,
            deviceId: String?,
            nCtx: Int,
            nGpuLayers: Int,
        ): NativeVlmSession {
            val dispatcher = Executors.newSingleThreadExecutor { runnable ->
                Thread(runnable, "unirt-vlm").apply { isDaemon = true }
            }.asCoroutineDispatcher()
            val handle = withContext(dispatcher) {
                Native.vlmCreate(modelPath, mmprojPath, pluginId, deviceId, nCtx, nGpuLayers)
            }
            if (handle == 0L) {
                dispatcher.close()
                throw UniRTException(-1, "cannot load $modelPath: ${Native.lastError()}")
            }
            return NativeVlmSession(handle, dispatcher)
        }
    }

    private fun requireOpen(): Long {
        check(handle != 0L) { "session is closed" }
        return handle
    }

    private fun raise(): Nothing = throw UniRTException(-1, Native.lastError())

    /** Unpacks [VlmGenerateOptions] into the native call once — see
     *  NativeLlmSession.nativeGenerate for why Native.vlmGenerate itself
     *  stays flat/scalar instead of taking an object across JNI. */
    private fun nativeGenerate(
        prompt: String,
        options: VlmGenerateOptions,
        onToken: TokenCallback?,
    ): LlmGenerateResult = Native.vlmGenerate(
        requireOpen(), prompt, options.maxTokens, options.temperature,
        options.topP, options.topK, options.seed,
        options.imagePaths.toTypedArray(), options.audioPaths.toTypedArray(),
        options.imageMaxLength, onToken,
    ) ?: raise()

    override suspend fun capabilities(): VlmCapabilities =
        withContext(dispatcher) { Native.vlmGetCapabilities(requireOpen()) ?: raise() }

    override suspend fun applyChatTemplate(
        messages: List<VlmChatMessage>,
        enableThinking: Boolean,
        grounding: Boolean,
    ): String = withContext(dispatcher) {
        Native.vlmApplyChatTemplate(
            requireOpen(),
            messages.map { it.role }.toTypedArray(),
            messages.map { m -> m.contents.map { it.wireType }.toTypedArray() }.toTypedArray(),
            messages.map { m -> m.contents.map { it.wireText }.toTypedArray() }.toTypedArray(),
            enableThinking,
            grounding,
        ) ?: raise()
    }

    override suspend fun generate(prompt: String, options: VlmGenerateOptions): String =
        withContext(dispatcher) { nativeGenerate(prompt, options, onToken = null).text }

    // See NativeLlmSession.stream: Completed/Error are emitted after
    // vlmGenerate() itself returns, using its real result — the C ABI has no
    // separate "done" callback.
    override fun stream(prompt: String, options: VlmGenerateOptions): Flow<LlmStreamResult> =
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
        val status = withContext(dispatcher) { Native.vlmReset(requireOpen()) }
        if (status < 0) throw UniRTException(status, Native.errorMessage(status))
    }

    override suspend fun runtimeStats(): RuntimeStats =
        withContext(dispatcher) { Native.vlmRuntimeStats(requireOpen()) ?: raise() }

    override fun close() {
        if (handle == 0L) return
        runBlocking(dispatcher) {
            Native.vlmDestroy(handle)
            handle = 0L
        }
        dispatcher.close()
    }
}

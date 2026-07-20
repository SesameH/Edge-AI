// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

/** Raw JNI surface. Application code should prefer [UniRT] and [LlmSession]. */
internal object Native {
    init {
        System.loadLibrary("unirt_jni")
    }

    @JvmStatic external fun init(): Int
    @JvmStatic external fun deinit(): Int
    @JvmStatic external fun version(): String
    @JvmStatic external fun lastError(): String
    @JvmStatic external fun errorMessage(code: Int): String
    @JvmStatic external fun pluginList(): Array<String>

    @JvmStatic external fun llmCreate(
        modelPath: String, pluginId: String, deviceId: String?, nCtx: Int, nGpuLayers: Int,
    ): Long
    @JvmStatic external fun llmDestroy(handle: Long): Int
    @JvmStatic external fun llmReset(handle: Long): Int
    @JvmStatic external fun llmApplyChatTemplate(
        handle: Long, roles: Array<String>, contents: Array<String>, addGenerationPrompt: Boolean,
    ): String?
    @JvmStatic external fun llmGenerate(
        handle: Long, prompt: String, maxTokens: Int, temperature: Float, topP: Float,
        topK: Int, seed: Int, onToken: TokenCallback?,
    ): String?
}

/** Receives streamed token pieces; return false to stop generation. */
fun interface TokenCallback {
    fun onToken(piece: String): Boolean
}

class UniRTException(val code: Int, detail: String) :
    RuntimeException("UniRT error $code: $detail")

private fun check(code: Int) {
    if (code < 0) throw UniRTException(code, Native.errorMessage(code) + ' ' + Native.lastError())
}

/** Runtime lifecycle. Call [start] once (e.g. in Application.onCreate). */
object UniRT {
    fun start() = check(Native.init())
    fun stop() = check(Native.deinit())
    fun version(): String = Native.version()
    fun plugins(): List<String> = Native.pluginList().toList()
}

data class ChatMessage(val role: String, val content: String)

/**
 * One loaded text model. Not thread-safe; drive a session from one thread
 * (or an actor/dispatcher). Close it before [UniRT.stop].
 */
class LlmSession(
    modelPath: String,
    pluginId: String = "llama_cpp",
    deviceId: String? = null,
    nCtx: Int = 0,
    nGpuLayers: Int = -1,
) : AutoCloseable {
    private var handle: Long = Native.llmCreate(modelPath, pluginId, deviceId, nCtx, nGpuLayers)

    init {
        if (handle == 0L) {
            throw UniRTException(-1, "cannot load $modelPath: ${Native.lastError()}")
        }
    }

    private fun requireOpen(): Long {
        require(handle != 0L) { "session is closed" }
        return handle
    }

    /** Render a conversation through the model's chat template. */
    fun applyChatTemplate(
        messages: List<ChatMessage>,
        addGenerationPrompt: Boolean = true,
    ): String =
        Native.llmApplyChatTemplate(
            requireOpen(),
            messages.map { it.role }.toTypedArray(),
            messages.map { it.content }.toTypedArray(),
            addGenerationPrompt,
        ) ?: throw UniRTException(-1, Native.lastError())

    /**
     * Generate from a rendered prompt. Resending a growing transcript reuses
     * the KV prefix automatically; call [reset] for an unrelated conversation.
     */
    fun generate(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0f,
        topP: Float = 0f,
        topK: Int = 0,
        seed: Int = 0,
        onToken: TokenCallback? = null,
    ): String =
        Native.llmGenerate(requireOpen(), prompt, maxTokens, temperature, topP, topK, seed, onToken)
            ?: throw UniRTException(-1, Native.lastError())

    /** Chat convenience: template + generate in one call. */
    fun chat(
        messages: List<ChatMessage>,
        maxTokens: Int = 512,
        temperature: Float = 0f,
        onToken: TokenCallback? = null,
    ): String = generate(
        applyChatTemplate(messages),
        maxTokens = maxTokens,
        temperature = temperature,
        onToken = onToken,
    )

    fun reset() = check(Native.llmReset(requireOpen()))

    override fun close() {
        if (handle != 0L) {
            Native.llmDestroy(handle)
            handle = 0L
        }
    }
}

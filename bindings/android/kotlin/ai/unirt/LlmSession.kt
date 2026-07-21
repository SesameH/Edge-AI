// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

import kotlinx.coroutines.flow.Flow

/** Timing, throughput, and stop cause for one generate() call — mirrors the
 *  Python binding's GenerationProfile (`unirt_ProfileData` subset it uses). */
data class GenerationProfile(
    val ttft: Long,
    val promptTime: Long,
    val decodeTime: Long,
    val promptTokens: Long,
    val generatedTokens: Long,
    val prefillSpeed: Double,
    val decodeSpeed: Double,
    val stopReason: String,
)

/** Raw JNI result of one llmGenerate() call: the full text plus its profile. */
data class LlmGenerateResult(val text: String, val profile: GenerationProfile)

/** One event from [LlmSession.stream]. */
sealed interface LlmStreamResult {
    data class Token(val text: String) : LlmStreamResult
    data class Completed(val profile: GenerationProfile) : LlmStreamResult
    data class Error(val cause: UniRTException) : LlmStreamResult
}

data class ChatMessage(val role: String, val content: String) {
    companion object {
        fun user(content: String) = ChatMessage("user", content)
        fun assistant(content: String) = ChatMessage("assistant", content)
        fun system(content: String) = ChatMessage("system", content)
    }
}

/** Sampling controls; the defaults mean greedy decoding. */
data class GenerateOptions(
    val maxTokens: Int = 512,
    val temperature: Float = 0f,
    val topP: Float = 0f,
    val topK: Int = 0,
    val seed: Int = 0,
)

class UniRTException(val code: Int, detail: String) :
    RuntimeException("UniRT error $code: $detail")

/**
 * One loaded text model. Obtain from [UniRT.createLlmSession]; every member
 * is safe to call from any coroutine — work is confined to the session's own
 * single-threaded dispatcher, matching the native handle's threading
 * contract. Close the session before [UniRT.stop].
 */
interface LlmSession : AutoCloseable {
    /** Render a conversation through the model's chat template. */
    suspend fun applyChatTemplate(
        messages: List<ChatMessage>,
        addGenerationPrompt: Boolean = true,
    ): String

    /** Generate to completion and return the full reply. */
    suspend fun generate(prompt: String, options: GenerateOptions = GenerateOptions()): String

    /** Generate as a cold [Flow] of [LlmStreamResult]: zero or more [LlmStreamResult.Token]
     *  followed by exactly one [LlmStreamResult.Completed] or [LlmStreamResult.Error].
     *  Cancelling the collector stops decoding. Resending a growing transcript reuses
     *  the KV prefix. */
    fun stream(prompt: String, options: GenerateOptions = GenerateOptions()): Flow<LlmStreamResult>

    /** Drop the conversation state (KV cache and transcript). */
    suspend fun reset()

    /** Template + generate in one call. */
    suspend fun chat(
        messages: List<ChatMessage>,
        options: GenerateOptions = GenerateOptions(),
    ): String = generate(applyChatTemplate(messages), options)
}

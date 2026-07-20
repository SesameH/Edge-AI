// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

import kotlinx.coroutines.flow.Flow

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

    /** Generate as a cold [Flow] of token pieces; cancelling the collector
     *  stops decoding. Resending a growing transcript reuses the KV prefix. */
    fun stream(prompt: String, options: GenerateOptions = GenerateOptions()): Flow<String>

    /** Drop the conversation state (KV cache and transcript). */
    suspend fun reset()

    /** Template + generate in one call. */
    suspend fun chat(
        messages: List<ChatMessage>,
        options: GenerateOptions = GenerateOptions(),
    ): String = generate(applyChatTemplate(messages), options)
}

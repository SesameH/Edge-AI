// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

import kotlinx.coroutines.flow.Flow

/** One piece of a multimodal turn: plain text, or a path to an image/audio file. */
sealed interface ContentPart {
    data class Text(val text: String) : ContentPart
    data class Image(val path: String) : ContentPart
    data class Audio(val path: String) : ContentPart
}

internal val ContentPart.wireType: String
    get() = when (this) {
        is ContentPart.Text -> "text"
        is ContentPart.Image -> "image"
        is ContentPart.Audio -> "audio"
    }

internal val ContentPart.wireText: String
    get() = when (this) {
        is ContentPart.Text -> text
        is ContentPart.Image -> path
        is ContentPart.Audio -> path
    }

data class VlmChatMessage(val role: String, val contents: List<ContentPart>) {
    companion object {
        fun user(vararg contents: ContentPart) = VlmChatMessage("user", contents.toList())
        fun user(text: String) = VlmChatMessage("user", listOf(ContentPart.Text(text)))
        fun assistant(text: String) = VlmChatMessage("assistant", listOf(ContentPart.Text(text)))
        fun system(text: String) = VlmChatMessage("system", listOf(ContentPart.Text(text)))
    }
}

/** Which media the loaded projector actually accepts (see [VlmSession.capabilities]). */
data class VlmCapabilities(val supportsVision: Boolean, val supportsAudio: Boolean)

/** Sampling controls plus per-request media. Kept separate from [GenerateOptions]
 *  rather than adding image/audio fields there: those fields would be dead
 *  weight on every LLM call, which never consumes them. */
data class VlmGenerateOptions(
    val maxTokens: Int = 512,
    val temperature: Float = 0f,
    val topP: Float = 0f,
    val topK: Int = 0,
    val seed: Int = 0,
    val imagePaths: List<String> = emptyList(),
    val audioPaths: List<String> = emptyList(),
    /** Cap on the longest image edge; 0 = no resize. */
    val imageMaxLength: Int = 0,
)

/**
 * One loaded multimodal model. Obtain from [UniRT.createVlmSession]; same
 * threading contract as [LlmSession] — every member is safe to call from any
 * coroutine, work is confined to the session's own dispatcher. Close the
 * session before [UniRT.stop].
 */
interface VlmSession : AutoCloseable {
    /** Which media the loaded projector accepts; llama_cpp reflects the
     *  mmproj, other plugins may report both false. */
    suspend fun capabilities(): VlmCapabilities

    /** Render a multimodal conversation through the model's chat template. */
    suspend fun applyChatTemplate(
        messages: List<VlmChatMessage>,
        enableThinking: Boolean = false,
        grounding: Boolean = false,
    ): String

    /** Generate to completion and return the full reply. */
    suspend fun generate(prompt: String, options: VlmGenerateOptions = VlmGenerateOptions()): String

    /** Generate as a cold [Flow] of [LlmStreamResult]; cancelling the collector
     *  stops decoding. */
    fun stream(prompt: String, options: VlmGenerateOptions = VlmGenerateOptions()): Flow<LlmStreamResult>

    /** Drop the conversation state (KV cache and transcript). */
    suspend fun reset()

    /** Template + generate in one call. */
    suspend fun chat(
        messages: List<VlmChatMessage>,
        options: VlmGenerateOptions = VlmGenerateOptions(),
    ): String = generate(applyChatTemplate(messages), options)
}

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.test.runTest

private class FakeVlmSession : VlmSession {
    val appliedMessages = mutableListOf<List<VlmChatMessage>>()
    val generatedPrompts = mutableListOf<String>()

    override suspend fun capabilities(): VlmCapabilities = VlmCapabilities(true, false)

    override suspend fun applyChatTemplate(
        messages: List<VlmChatMessage>,
        enableThinking: Boolean,
        grounding: Boolean,
    ): String {
        appliedMessages += messages
        return messages.joinToString("\n") { m ->
            "${m.role}: " + m.contents.joinToString(",") { it.wireText }
        }
    }

    override suspend fun generate(prompt: String, options: VlmGenerateOptions): String {
        generatedPrompts += prompt
        return "reply to: $prompt"
    }

    override fun stream(prompt: String, options: VlmGenerateOptions): Flow<LlmStreamResult> =
        throw NotImplementedError("not exercised by this test")

    override suspend fun reset() {}

    override suspend fun runtimeStats(): RuntimeStats =
        RuntimeStats(-1, -1, -1, -1, "fake")

    override fun close() {}
}

class VlmSessionTest {
    @Test
    fun chatRendersTheTemplateThenGeneratesFromIt() = runTest {
        val session = FakeVlmSession()
        val message = VlmChatMessage.user(
            ContentPart.Text("what is this?"),
            ContentPart.Image("/tmp/photo.jpg"),
        )

        val reply = session.chat(listOf(message))

        assertEquals(listOf(listOf(message)), session.appliedMessages)
        assertEquals(listOf("user: what is this?,/tmp/photo.jpg"), session.generatedPrompts)
        assertEquals("reply to: user: what is this?,/tmp/photo.jpg", reply)
    }
}

class ContentPartTest {
    @Test
    fun wireTypeAndTextMapEachContentKind() {
        assertEquals("text", ContentPart.Text("hi").wireType)
        assertEquals("hi", ContentPart.Text("hi").wireText)
        assertEquals("image", ContentPart.Image("/a.jpg").wireType)
        assertEquals("/a.jpg", ContentPart.Image("/a.jpg").wireText)
        assertEquals("audio", ContentPart.Audio("/a.wav").wireType)
        assertEquals("/a.wav", ContentPart.Audio("/a.wav").wireText)
    }
}

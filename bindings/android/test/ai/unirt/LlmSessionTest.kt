// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.test.runTest

/** LlmSession is an interface specifically so it's fakeable here — see
 *  Native.kt's System.loadLibrary comment for why the real implementation
 *  cannot run in a plain JVM test. */
private class FakeLlmSession : LlmSession {
    val appliedMessages = mutableListOf<List<ChatMessage>>()
    val generatedPrompts = mutableListOf<String>()

    override suspend fun applyChatTemplate(
        messages: List<ChatMessage>,
        addGenerationPrompt: Boolean,
    ): String {
        appliedMessages += messages
        return messages.joinToString("\n") { "${it.role}: ${it.content}" }
    }

    override suspend fun generate(prompt: String, options: GenerateOptions): String {
        generatedPrompts += prompt
        return "reply to: $prompt"
    }

    override fun stream(prompt: String, options: GenerateOptions): Flow<LlmStreamResult> =
        throw NotImplementedError("not exercised by this test")

    override suspend fun reset() {}

    override fun close() {}
}

class LlmSessionTest {
    @Test
    fun chatRendersTheTemplateThenGeneratesFromIt() = runTest {
        val session = FakeLlmSession()

        val reply = session.chat(listOf(ChatMessage.user("hi")))

        assertEquals(listOf(listOf(ChatMessage.user("hi"))), session.appliedMessages)
        assertEquals(listOf("user: hi"), session.generatedPrompts)
        assertEquals("reply to: user: hi", reply)
    }
}

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.test.runTest

/** Replays a canned reply so the tool plumbing can be exercised without a
 *  model — see LlmSessionTest for why LlmSession is an interface. */
private class ScriptedLlmSession(private val reply: String) : LlmSession {
    var lastPrompt: String? = null
    var lastOptions: GenerateOptions? = null

    override suspend fun applyChatTemplate(
        messages: List<ChatMessage>,
        addGenerationPrompt: Boolean,
    ): String = messages.joinToString("\n") { "${it.role}: ${it.content}" }

    override suspend fun generate(prompt: String, options: GenerateOptions): String {
        lastPrompt = prompt
        lastOptions = options
        return reply
    }

    override fun stream(prompt: String, options: GenerateOptions): Flow<LlmStreamResult> =
        throw NotImplementedError("not exercised by this test")

    override suspend fun reset() {}

    override suspend fun runtimeStats(): RuntimeStats = RuntimeStats(-1, -1, -1, -1, "fake")

    override fun close() {}
}

private val weather = ToolDefinition(
    name = "get_weather",
    description = "Look up the weather",
    parametersJson = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""",
)

private val ping = ToolDefinition(name = "ping")

class ToolCallingTest {
    @Test
    fun autoSchemaOffersOneBranchPerToolPlusText() {
        val plan = ToolPlan.of(listOf(weather, ping))!!

        // Byte-for-byte what the Swift binding emits for the same tools, and
        // structurally what the Python binding emits — the three bindings are
        // meant to constrain a model identically. The caller's parameter
        // schema appears verbatim, property order intact, and the no-argument
        // tool still gets a value the grammar can accept.
        assertEquals(
            """{"oneOf":[""" +
                """{"type":"object","properties":{"name":{"const":"get_weather"},""" +
                """"arguments":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},""" +
                """"required":["name","arguments"],"additionalProperties":false},""" +
                """{"type":"object","properties":{"name":{"const":"ping"},""" +
                """"arguments":{"type":"object","properties":{},"additionalProperties":false}},""" +
                """"required":["name","arguments"],"additionalProperties":false},""" +
                """{"type":"object","properties":{"content":{"type":"string"}},""" +
                """"required":["content"],"additionalProperties":false}]}""",
            plan.schemaJson,
        )
    }

    @Test
    fun requiredDropsTheTextBranch() {
        val schema = ToolPlan.of(listOf(weather, ping), ToolChoice.Required)!!.schemaJson

        assertTrue(schema.contains(""""const":"get_weather""""), schema)
        assertTrue(!schema.contains(""""content""""), schema)
    }

    @Test
    fun namingOneToolCollapsesToThatSingleBranch() {
        val schema = ToolPlan.of(listOf(weather, ping), ToolChoice.Function("ping"))!!.schemaJson

        // One branch means no oneOf wrapper at all.
        assertTrue(schema.startsWith("""{"type":"object""""), schema)
        assertTrue(schema.contains(""""const":"ping""""), schema)
        assertTrue(!schema.contains("get_weather"), schema)
    }

    @Test
    fun noneMeansNoPlan() {
        assertNull(ToolPlan.of(listOf(weather), ToolChoice.None))
    }

    @Test
    fun theSystemPromptDescribesEveryToolAndBothReplyShapes() {
        val prompt = ToolPlan.of(listOf(weather, ping))!!.systemPrompt

        assertEquals(
            """
            You have access to these tools:

            - get_weather: Look up the weather
              parameters: {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
            - ping: no description provided
              parameters: {"type":"object","properties":{},"additionalProperties":false}

            Reply with a single JSON object and nothing else. To call a tool, use {"name": "<tool name>", "arguments": {...}}.
            To answer the user directly instead, use {"content": "<your reply>"}.
            """.trimIndent(),
            prompt,
        )
    }

    @Test
    fun requiredPromptOmitsTheDirectAnswerLine() {
        val prompt = ToolPlan.of(listOf(ping), ToolChoice.Required)!!.systemPrompt

        assertTrue(!prompt.contains("To answer the user directly"), prompt)
    }

    @Test
    fun rejectsEmptyDuplicateAndUndeclaredTools() {
        assertFailsWith<IllegalArgumentException> { ToolPlan.of(emptyList()) }
        assertFailsWith<IllegalArgumentException> { ToolPlan.of(listOf(ping, ping)) }
        assertFailsWith<IllegalArgumentException> { ToolPlan.of(listOf(ping.copy(name = ""))) }
        assertFailsWith<IllegalArgumentException> {
            ToolPlan.of(listOf(ping), ToolChoice.Function("nope"))
        }
    }

    @Test
    fun aCallIsReturnedWithItsArgumentsVerbatim() {
        val plan = ToolPlan.of(listOf(weather))!!

        val reply = plan.interpret("""{"name": "get_weather", "arguments": {"city": "Taipei"}}""")

        val call = (reply as ToolReply.Call).call
        assertEquals("get_weather", call.name)
        assertEquals("""{"city": "Taipei"}""", call.argumentsJson)
        assertTrue(call.id.startsWith("call_"), call.id)
    }

    @Test
    fun theTextBranchIsUnescapedBackToPlainText() {
        val plan = ToolPlan.of(listOf(weather))!!

        val reply = plan.interpret("""{"content": "line\none \"quoted\" é"}""")

        assertEquals("line\none \"quoted\" é", (reply as ToolReply.Text).content)
    }

    @Test
    fun anEscapedSurrogatePairDecodesToTheCharacterItEncodes() {
        val plan = ToolPlan.of(listOf(weather))!!

        // Assembled from the backslash so the source really carries the two
        // six-character escapes a model would emit, rather than the emoji
        // itself — that is the case the UTF-16 decoding path exists for.
        val backslash = Char(92)
        val pair = "${backslash}ud83d${backslash}ude00"

        val escaped = plan.interpret("""{"content": "a $pair b"}""")
        val literal = plan.interpret("""{"content": "a 😀 b"}""")

        assertEquals("a 😀 b", (escaped as ToolReply.Text).content)
        assertEquals(escaped.content, (literal as ToolReply.Text).content)
    }

    @Test
    fun aTruncatedReplySurfacesAsTextRatherThanAFabricatedCall() {
        val plan = ToolPlan.of(listOf(weather))!!

        // What maxTokens cutting a call mid-arguments looks like.
        val cut = """{"name": "get_weather", "arguments": {"city": "Tai"""
        assertEquals(cut, (plan.interpret(cut) as ToolReply.Text).content)
    }

    @Test
    fun anUndeclaredToolNameIsNotTrusted() {
        val plan = ToolPlan.of(listOf(weather))!!

        val text = """{"name": "rm_rf", "arguments": {}}"""
        assertEquals(text, (plan.interpret(text) as ToolReply.Text).content)
    }

    @Test
    fun chatWithToolsPrependsTheInstructionsAndSpendsTheSchemaSlot() = runTest {
        val session = ScriptedLlmSession("""{"name": "ping", "arguments": {}}""")

        val reply = session.chatWithTools(listOf(ChatMessage.user("hi")), listOf(ping))

        assertEquals("ping", (reply as ToolReply.Call).call.name)
        assertTrue(session.lastPrompt!!.startsWith("system: You have access to these tools:"))
        assertTrue(session.lastPrompt!!.endsWith("user: hi"))
        assertEquals(ToolPlan.of(listOf(ping))!!.schemaJson, session.lastOptions!!.jsonSchema)
    }

    @Test
    fun chatWithToolsRefusesToShareTheGrammarSlot() = runTest {
        val session = ScriptedLlmSession("{}")

        assertFailsWith<IllegalArgumentException> {
            session.chatWithTools(
                listOf(ChatMessage.user("hi")),
                listOf(ping),
                options = GenerateOptions(jsonMode = true),
            )
        }
    }

    @Test
    fun toolChoiceNoneFallsBackToAPlainTurn() = runTest {
        val session = ScriptedLlmSession("just text")

        val reply = session.chatWithTools(
            listOf(ChatMessage.user("hi")), listOf(ping), ToolChoice.None
        )

        assertEquals("just text", (reply as ToolReply.Text).content)
        assertNull(session.lastOptions!!.jsonSchema)
        assertTrue(!session.lastPrompt!!.contains("You have access to these tools"))
    }

    @Test
    fun historyTurnsReplayACallAsTheJsonTheGrammarProduced() {
        val call = ToolCall("call_1", "get_weather", """{"city": "Taipei"}""")

        assertEquals(
            ChatMessage.assistant("""{"name": "get_weather", "arguments": {"city": "Taipei"}}"""),
            ChatMessage.toolCall(call),
        )
        assertEquals(
            ChatMessage.user("Result from get_weather: 21C"),
            ChatMessage.toolResult("get_weather", "21C"),
        )
        assertEquals(ChatMessage.user("Tool result: 21C"), ChatMessage.toolResult(null, "21C"))
    }
}

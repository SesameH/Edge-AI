// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

import java.util.UUID

/**
 * OpenAI-style tool calling, built on grammar-constrained decoding — the
 * Kotlin counterpart of the Python binding's `unirt/tool_calling.py`, and it
 * behaves the same way for the same reasons.
 *
 * A hosted API can let a model emit a proprietary tool-call token sequence. A
 * local 1B model cannot be trusted to invent one, so this spends the grammar
 * slot instead: the declared tools are compiled into a JSON schema the sampler
 * physically cannot leave. A reply therefore always parses, always names a
 * declared tool, and always carries arguments that validate against that
 * tool's own parameter schema — there is no "the model returned malformed
 * JSON" failure mode to retry around.
 *
 * Two deliberate narrowings, both inherited from the Python implementation.
 * Only one call comes back per turn: parallel calls need the model to plan a
 * whole batch before seeing any result, which is what small models are worst
 * at. And the tool definitions are rendered into a system message rather than
 * passed to the chat template, because `llama_chat_apply_template` has no
 * tools parameter and the MLX backend has no tool-aware template either —
 * doing it in the prompt is the only thing that behaves identically on both.
 *
 * One difference from Python worth knowing: [ToolDefinition.parametersJson] is
 * caller-supplied JSON Schema *text* that is embedded verbatim, never parsed
 * and re-serialized. That keeps property order exactly as written (which the
 * MLX constraint is sensitive to) and keeps this file free of a JSON
 * dependency, at the cost of the prompt echoing the caller's own whitespace.
 */

/** One tool the model may call.
 *
 *  @param parametersJson JSON Schema text for the arguments object, embedded
 *    verbatim into the generated schema. `null` means the tool takes no
 *    arguments, which still has to produce *something* the grammar accepts —
 *    the empty object. */
data class ToolDefinition(
    val name: String,
    val description: String? = null,
    val parametersJson: String? = null,
)

/** Which branches of the generated schema the model is allowed to take. */
sealed interface ToolChoice {
    /** One text branch plus one branch per tool. */
    data object Auto : ToolChoice

    /** Tools are dropped entirely; a plain text turn. */
    data object None : ToolChoice

    /** Tool branches only, so the model must call something. */
    data object Required : ToolChoice

    /** That one tool's branch alone. */
    data class Function(val name: String) : ToolChoice
}

/** One tool invocation the model asked for. [argumentsJson] is a JSON object
 *  that validates against the tool's own parameter schema. */
data class ToolCall(val id: String, val name: String, val argumentsJson: String)

/** The outcome of a tool-enabled turn: the model either answered or called. */
sealed interface ToolReply {
    data class Text(val content: String) : ToolReply
    data class Call(val call: ToolCall) : ToolReply
}

/** Everything a tool-enabled turn needs, derived from the declared tools.
 *  Build with [ToolPlan.of]; `null` there means this turn uses no tools. */
class ToolPlan private constructor(
    val tools: List<ToolDefinition>,
    private val forcedName: String?,
    private val mustCall: Boolean,
) {
    /** The JSON schema that constrains this turn's output. */
    val schemaJson: String by lazy {
        val branches = selectedTools.map { tool ->
            """{"type":"object","properties":{"name":{"const":${jsonString(tool.name)}},""" +
                """"arguments":${tool.parametersJson ?: EMPTY_PARAMETERS}},""" +
                """"required":["name","arguments"],"additionalProperties":false}"""
        }.toMutableList()
        if (!mustCall) {
            branches += """{"type":"object","properties":{"$TEXT_BRANCH_KEY":{"type":"string"}},""" +
                """"required":["$TEXT_BRANCH_KEY"],"additionalProperties":false}"""
        }
        if (branches.size == 1) branches[0] else """{"oneOf":[${branches.joinToString(",")}]}"""
    }

    /** Instructions describing the tools and the reply format. */
    val systemPrompt: String by lazy {
        buildString {
            append("You have access to these tools:\n\n")
            for (tool in selectedTools) {
                append("- ${tool.name}: ${tool.description ?: "no description provided"}\n")
                append("  parameters: ${tool.parametersJson ?: EMPTY_PARAMETERS}\n")
            }
            append("\n")
            append(
                "Reply with a single JSON object and nothing else. To call a tool, use " +
                    """{"name": "<tool name>", "arguments": {...}}."""
            )
            if (!mustCall) {
                append(
                    "\nTo answer the user directly instead, use " +
                        """{"$TEXT_BRANCH_KEY": "<your reply>"}."""
                )
            }
        }
    }

    private val selectedTools: List<ToolDefinition>
        get() = if (forcedName == null) tools else tools.filter { it.name == forcedName }

    /**
     * Split constrained output into a reply.
     *
     * The grammar guarantees the shape, so anything unparseable here means
     * generation was cut short by maxTokens; that surfaces as plain text
     * rather than a fabricated call, since a truncated call is not one the
     * caller can safely execute.
     */
    fun interpret(text: String): ToolReply {
        val fields = parseTopLevelObject(text) ?: return ToolReply.Text(text)
        val rawContent = fields[TEXT_BRANCH_KEY]
        if (rawContent != null && !fields.containsKey("name")) {
            return ToolReply.Text(decodeJsonString(rawContent) ?: text)
        }
        val name = fields["name"]?.let { decodeJsonString(it) }
        if (name == null || tools.none { it.name == name }) return ToolReply.Text(text)
        return ToolReply.Call(
            ToolCall(
                id = "call_" + UUID.randomUUID().toString().replace("-", "").take(24),
                name = name,
                argumentsJson = fields["arguments"] ?: "{}",
            )
        )
    }

    companion object {
        /** Validate [tools] against [choice]; `null` when this turn uses no
         *  tools (which is only [ToolChoice.None] — an empty tool list is a
         *  caller mistake, not a way to disable them). */
        fun of(tools: List<ToolDefinition>, choice: ToolChoice = ToolChoice.Auto): ToolPlan? {
            require(tools.isNotEmpty()) { "tools must be a non-empty list" }
            val seen = mutableSetOf<String>()
            for (tool in tools) {
                // NUL is rejected rather than escaped: these strings reach the
                // model through a NUL-terminated C string, so one embedded
                // here would silently truncate the prompt instead of failing.
                require(tool.name.isNotEmpty() && !tool.name.contains('\u0000')) {
                    "each tool needs a non-empty NUL-free name"
                }
                require(tool.description?.contains('\u0000') != true) {
                    "tool description must be NUL-free"
                }
                require(seen.add(tool.name)) { "duplicate tool name: ${tool.name}" }
            }
            return when (choice) {
                is ToolChoice.None -> null
                is ToolChoice.Auto -> ToolPlan(tools, null, false)
                is ToolChoice.Required -> ToolPlan(tools, null, true)
                is ToolChoice.Function -> {
                    require(choice.name in seen) {
                        "tool choice names an undeclared tool: ${choice.name}"
                    }
                    ToolPlan(tools, choice.name, true)
                }
            }
        }
    }
}

/**
 * Template + generate one tool-enabled turn.
 *
 * The tool instructions ride in front of [messages] as a system turn and the
 * generated schema takes the grammar slot, so [options] must not already
 * constrain decoding — one turn has one grammar.
 */
suspend fun LlmSession.chatWithTools(
    messages: List<ChatMessage>,
    tools: List<ToolDefinition>,
    toolChoice: ToolChoice = ToolChoice.Auto,
    options: GenerateOptions = GenerateOptions(),
): ToolReply {
    val plan = ToolPlan.of(tools, toolChoice)
        ?: return ToolReply.Text(chat(messages, options))
    require(options.grammar == null && !options.jsonMode && options.jsonSchema == null) {
        "tools and grammar/jsonMode/jsonSchema cannot both constrain one turn"
    }
    val prompt = applyChatTemplate(listOf(ChatMessage.system(plan.systemPrompt)) + messages)
    return plan.interpret(generate(prompt, options.copy(jsonSchema = plan.schemaJson)))
}

/**
 * The assistant turn to append after the model called a tool.
 *
 * Chat templates reached through `llama_chat_apply_template` only handle
 * system/user/assistant roles, and an unknown role there is a hard template
 * error rather than a degraded prompt. So a prior call is replayed as the
 * exact JSON the grammar had produced — the transcript the model reads back
 * matches what it was constrained to write.
 */
fun ChatMessage.Companion.toolCall(call: ToolCall): ChatMessage =
    ChatMessage.assistant("""{"name": ${jsonString(call.name)}, "arguments": ${call.argumentsJson}}""")

/** The turn to append after running a tool. It is a *user* turn because that
 *  is the only role every chat template accepts. */
fun ChatMessage.Companion.toolResult(name: String?, content: String): ChatMessage =
    ChatMessage.user(if (name != null) "Result from $name: $content" else "Tool result: $content")

// Wrapping assistant text in JSON costs escaping and a little fluency, so the
// text branch is described to the model in the same breath as the tools.
private const val TEXT_BRANCH_KEY = "content"

// A no-argument tool still has to produce *something* the grammar can accept,
// and `{}` is the JSON that means "no arguments".
private const val EMPTY_PARAMETERS = """{"type":"object","properties":{},"additionalProperties":false}"""

/** Quote and escape [value] as a JSON string literal. */
internal fun jsonString(value: String): String = buildString {
    append('"')
    for (ch in value) {
        when {
            ch == '"' -> append("\\\"")
            ch == '\\' -> append("\\\\")
            ch == '\n' -> append("\\n")
            ch == '\r' -> append("\\r")
            ch == '\t' -> append("\\t")
            ch < ' ' -> append("\\u%04x".format(ch.code))
            else -> append(ch)
        }
    }
    append('"')
}

/**
 * Map the members of a top-level JSON object to the raw source text of their
 * values, or `null` if [text] is not one complete object.
 *
 * Values stay as source slices rather than becoming a parsed tree: `arguments`
 * is handed back to the caller verbatim, so nothing here needs to understand
 * numbers, nesting or unicode escapes — only where each value ends.
 */
internal fun parseTopLevelObject(text: String): Map<String, String>? {
    var i = skipWhitespace(text, 0)
    if (i >= text.length || text[i] != '{') return null
    i++
    val fields = LinkedHashMap<String, String>()
    i = skipWhitespace(text, i)
    if (i < text.length && text[i] == '}') return if (skipWhitespace(text, i + 1) == text.length) fields else null
    while (true) {
        i = skipWhitespace(text, i)
        if (i >= text.length || text[i] != '"') return null
        val keyEnd = scanValue(text, i) ?: return null
        val key = decodeJsonString(text.substring(i, keyEnd)) ?: return null
        i = skipWhitespace(text, keyEnd)
        if (i >= text.length || text[i] != ':') return null
        i = skipWhitespace(text, i + 1)
        val valueEnd = scanValue(text, i) ?: return null
        fields[key] = text.substring(i, valueEnd)
        i = skipWhitespace(text, valueEnd)
        if (i >= text.length) return null
        when (text[i]) {
            ',' -> i++
            '}' -> return if (skipWhitespace(text, i + 1) == text.length) fields else null
            else -> return null
        }
    }
}

/** Decode a JSON string literal, or `null` if [raw] is not one. */
internal fun decodeJsonString(raw: String): String? {
    if (raw.length < 2 || raw[0] != '"' || raw[raw.length - 1] != '"') return null
    val out = StringBuilder()
    var i = 1
    val end = raw.length - 1
    while (i < end) {
        val ch = raw[i]
        if (ch != '\\') {
            out.append(ch)
            i++
            continue
        }
        i++
        if (i >= end) return null
        when (val escape = raw[i]) {
            '"', '\\', '/' -> out.append(escape)
            'b' -> out.append('\b')
            'f' -> out.append('\u000C')
            'n' -> out.append('\n')
            'r' -> out.append('\r')
            't' -> out.append('\t')
            'u' -> {
                if (i + 5 > end) return null
                val hex = raw.substring(i + 1, i + 5)
                out.append((hex.toIntOrNull(16) ?: return null).toChar())
                i += 4
            }
            else -> return null
        }
        i++
    }
    return out.toString()
}

/** Index just past the JSON value starting at [start], or `null` if it is
 *  malformed or runs off the end (which is what a truncated reply looks like). */
private fun scanValue(text: String, start: Int): Int? {
    if (start >= text.length) return null
    when (text[start]) {
        '"' -> {
            var i = start + 1
            while (i < text.length) {
                when (text[i]) {
                    '\\' -> i += 2
                    '"' -> return i + 1
                    else -> i++
                }
            }
            return null
        }
        '{', '[' -> {
            var depth = 0
            var i = start
            while (i < text.length) {
                when (text[i]) {
                    '{', '[' -> depth++
                    '}', ']' -> {
                        depth--
                        if (depth == 0) return i + 1
                    }
                    '"' -> i = (scanValue(text, i) ?: return null) - 1
                }
                i++
            }
            return null
        }
        else -> {
            var i = start
            while (i < text.length && text[i] !in ",}] \t\n\r") i++
            return if (i == start) null else i
        }
    }
}

private fun skipWhitespace(text: String, start: Int): Int {
    var i = start
    while (i < text.length && text[i].isWhitespace()) i++
    return i
}

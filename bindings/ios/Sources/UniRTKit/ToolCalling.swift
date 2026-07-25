// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

/// OpenAI-style tool calling, built on grammar-constrained decoding — the
/// Swift counterpart of the Python binding's `unirt/tool_calling.py` and the
/// Kotlin binding's `ToolCalling.kt`, and it behaves the same way for the same
/// reasons.
///
/// A hosted API can let a model emit a proprietary tool-call token sequence. A
/// local 1B model cannot be trusted to invent one, so this spends the grammar
/// slot instead: the declared tools are compiled into a JSON schema the sampler
/// physically cannot leave. A reply therefore always parses, always names a
/// declared tool, and always carries arguments that validate against that
/// tool's own parameter schema — there is no "the model returned malformed
/// JSON" failure mode to retry around.
///
/// Two deliberate narrowings, both inherited from the Python implementation.
/// Only one call comes back per turn: parallel calls need the model to plan a
/// whole batch before seeing any result, which is what small models are worst
/// at. And the tool definitions are rendered into a system message rather than
/// passed to the chat template, because `llama_chat_apply_template` has no
/// tools parameter and the MLX backend has no tool-aware template either —
/// doing it in the prompt is the only thing that behaves identically on both.
///
/// One difference from Python worth knowing: ``ToolDefinition/parametersJson``
/// is caller-supplied JSON Schema *text* that is embedded verbatim, never
/// parsed and re-serialized. That keeps property order exactly as written
/// (which the MLX constraint is sensitive to) and keeps this file free of
/// Foundation, at the cost of the prompt echoing the caller's own whitespace.

/// One tool the model may call.
public struct ToolDefinition: Sendable, Equatable {
    public let name: String
    public let description: String?
    /// JSON Schema text for the arguments object, embedded verbatim into the
    /// generated schema. `nil` means the tool takes no arguments, which still
    /// has to produce *something* the grammar accepts — the empty object.
    public let parametersJson: String?

    public init(name: String, description: String? = nil, parametersJson: String? = nil) {
        self.name = name
        self.description = description
        self.parametersJson = parametersJson
    }
}

/// Which branches of the generated schema the model is allowed to take.
public enum ToolChoice: Sendable, Equatable {
    /// One text branch plus one branch per tool.
    case auto
    /// Tools are dropped entirely; a plain text turn.
    case none
    /// Tool branches only, so the model must call something.
    case required
    /// That one tool's branch alone.
    case function(name: String)
}

/// One tool invocation the model asked for. `argumentsJson` is a JSON object
/// that validates against the tool's own parameter schema.
public struct ToolCall: Sendable, Equatable {
    public let id: String
    public let name: String
    public let argumentsJson: String
}

/// The outcome of a tool-enabled turn: the model either answered or called.
public enum ToolReply: Sendable, Equatable {
    case text(String)
    case call(ToolCall)
}

/// Why a set of tool declarations was rejected.
public enum ToolCallingError: Error, Equatable, CustomStringConvertible {
    case noTools
    case invalidToolName(String)
    case duplicateToolName(String)
    case invalidToolDescription(String)
    case undeclaredTool(String)
    case grammarSlotTaken

    public var description: String {
        switch self {
        case .noTools:
            return "tools must be a non-empty list"
        case .invalidToolName(let name):
            return "each tool needs a non-empty NUL-free name (got \(name.debugDescription))"
        case .duplicateToolName(let name):
            return "duplicate tool name: \(name)"
        case .invalidToolDescription(let name):
            return "tool description must be NUL-free (tool \(name))"
        case .undeclaredTool(let name):
            return "tool choice names an undeclared tool: \(name)"
        case .grammarSlotTaken:
            return "tools and grammar/jsonMode/jsonSchema cannot both constrain one turn"
        }
    }
}

/// Everything a tool-enabled turn needs, derived from the declared tools.
/// Build with ``ToolPlan/make(tools:choice:)``; `nil` there means this turn
/// uses no tools.
public struct ToolPlan: Sendable {
    public let tools: [ToolDefinition]
    private let forcedName: String?
    private let mustCall: Bool

    /// Validate `tools` against `choice`; `nil` when this turn uses no tools
    /// (which is only ``ToolChoice/none`` — an empty tool list is a caller
    /// mistake, not a way to disable them).
    public static func make(tools: [ToolDefinition], choice: ToolChoice = .auto) throws -> ToolPlan? {
        guard !tools.isEmpty else { throw ToolCallingError.noTools }
        var seen = Set<String>()
        for tool in tools {
            // NUL is rejected rather than escaped: these strings reach the
            // model through a NUL-terminated C string, so one embedded here
            // would silently truncate the prompt instead of failing.
            guard !tool.name.isEmpty, !tool.name.unicodeScalars.contains("\u{0000}") else {
                throw ToolCallingError.invalidToolName(tool.name)
            }
            if let description = tool.description, description.unicodeScalars.contains("\u{0000}") {
                throw ToolCallingError.invalidToolDescription(tool.name)
            }
            guard seen.insert(tool.name).inserted else {
                throw ToolCallingError.duplicateToolName(tool.name)
            }
        }
        switch choice {
        case .none:
            return nil
        case .auto:
            return ToolPlan(tools: tools, forcedName: nil, mustCall: false)
        case .required:
            return ToolPlan(tools: tools, forcedName: nil, mustCall: true)
        case .function(let name):
            guard seen.contains(name) else { throw ToolCallingError.undeclaredTool(name) }
            return ToolPlan(tools: tools, forcedName: name, mustCall: true)
        }
    }

    private var selectedTools: [ToolDefinition] {
        guard let forcedName else { return tools }
        return tools.filter { $0.name == forcedName }
    }

    /// The JSON schema that constrains this turn's output.
    public var schemaJson: String {
        var branches = selectedTools.map { tool in
            #"{"type":"object","properties":{"name":{"const":"# + jsonString(tool.name)
                + #"},"arguments":"# + (tool.parametersJson ?? emptyParameters)
                + #"},"required":["name","arguments"],"additionalProperties":false}"#
        }
        if !mustCall {
            branches.append(
                #"{"type":"object","properties":{"\#(textBranchKey)":{"type":"string"}},"#
                    + #""required":["\#(textBranchKey)"],"additionalProperties":false}"#
            )
        }
        if branches.count == 1 { return branches[0] }
        return #"{"oneOf":["# + branches.joined(separator: ",") + "]}"
    }

    /// Instructions describing the tools and the reply format.
    public var systemPrompt: String {
        var text = "You have access to these tools:\n\n"
        for tool in selectedTools {
            text += "- \(tool.name): \(tool.description ?? "no description provided")\n"
            text += "  parameters: \(tool.parametersJson ?? emptyParameters)\n"
        }
        text += "\n"
        text += "Reply with a single JSON object and nothing else. To call a tool, use "
        text += #"{"name": "<tool name>", "arguments": {...}}."#
        if !mustCall {
            text += "\nTo answer the user directly instead, use "
            text += #"{"\#(textBranchKey)": "<your reply>"}."#
        }
        return text
    }

    /// Split constrained output into a reply.
    ///
    /// The grammar guarantees the shape, so anything unparseable here means
    /// generation was cut short by `maxTokens`; that surfaces as plain text
    /// rather than a fabricated call, since a truncated call is not one the
    /// caller can safely execute.
    public func interpret(_ text: String) -> ToolReply {
        guard let fields = parseTopLevelObject(text) else { return .text(text) }
        if let rawContent = fields[textBranchKey], fields["name"] == nil {
            return .text(decodeJsonString(rawContent) ?? text)
        }
        guard let rawName = fields["name"], let name = decodeJsonString(rawName),
              tools.contains(where: { $0.name == name })
        else { return .text(text) }
        return .call(
            ToolCall(
                id: "call_" + newCallSuffix(),
                name: name,
                argumentsJson: fields["arguments"] ?? "{}"
            )
        )
    }
}

extension LlmSession {
    /// Template + generate one tool-enabled turn.
    ///
    /// The tool instructions ride in front of `messages` as a system turn and
    /// the generated schema takes the grammar slot, so `options` must not
    /// already constrain decoding — one turn has one grammar.
    public func chatWithTools(
        _ messages: [ChatMessage],
        tools: [ToolDefinition],
        toolChoice: ToolChoice = .auto,
        options: GenerateOptions = GenerateOptions()
    ) throws -> ToolReply {
        guard let plan = try ToolPlan.make(tools: tools, choice: toolChoice) else {
            return .text(try chat(messages, options: options))
        }
        guard options.grammar == nil, !options.jsonMode, options.jsonSchema == nil else {
            throw ToolCallingError.grammarSlotTaken
        }
        var constrained = options
        constrained.jsonSchema = plan.schemaJson
        let prompt = try applyChatTemplate([.system(plan.systemPrompt)] + messages)
        return plan.interpret(try generate(prompt: prompt, options: constrained))
    }
}

extension ChatMessage {
    /// The assistant turn to append after the model called a tool.
    ///
    /// Chat templates reached through `llama_chat_apply_template` only handle
    /// system/user/assistant roles, and an unknown role there is a hard
    /// template error rather than a degraded prompt. So a prior call is
    /// replayed as the exact JSON the grammar had produced — the transcript
    /// the model reads back matches what it was constrained to write.
    public static func toolCall(_ call: ToolCall) -> ChatMessage {
        .assistant(
            #"{"name": "# + jsonString(call.name)
                + #", "arguments": "# + call.argumentsJson + "}"
        )
    }

    /// The turn to append after running a tool. It is a *user* turn because
    /// that is the only role every chat template accepts.
    public static func toolResult(name: String?, content: String) -> ChatMessage {
        .user(name.map { "Result from \($0): \(content)" } ?? "Tool result: \(content)")
    }
}

// Wrapping assistant text in JSON costs escaping and a little fluency, so the
// text branch is described to the model in the same breath as the tools.
private let textBranchKey = "content"

// A no-argument tool still has to produce *something* the grammar can accept,
// and `{}` is the JSON that means "no arguments".
private let emptyParameters = #"{"type":"object","properties":{},"additionalProperties":false}"#

/// 24 hex digits of call id, matching the Python binding's `uuid4().hex[:24]`.
/// The id only has to be unique within one conversation, so it is not drawn
/// from a cryptographic source.
private func newCallSuffix() -> String {
    String((0..<24).map { _ in hexDigits.randomElement()! })
}

/// Quote and escape `value` as a JSON string literal.
func jsonString(_ value: String) -> String {
    var out = "\""
    for scalar in value.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default:
            if scalar.value < 0x20 {
                // Every remaining control scalar is below 0x20, so the high
                // byte of the escape is always "00".
                out += "\\u00"
                out.append(hexDigits[Int(scalar.value) >> 4])
                out.append(hexDigits[Int(scalar.value) & 0xF])
            } else {
                out.unicodeScalars.append(scalar)
            }
        }
    }
    return out + "\""
}

private let hexDigits = Array("0123456789abcdef")

/// Map the members of a top-level JSON object to the raw source text of their
/// values, or `nil` if `text` is not one complete object.
///
/// Values stay as source slices rather than becoming a parsed tree:
/// `arguments` is handed back to the caller verbatim, so nothing here needs to
/// understand numbers, nesting or unicode escapes — only where each value ends.
func parseTopLevelObject(_ text: String) -> [String: String]? {
    let chars = Array(text)
    var i = skipWhitespace(chars, 0)
    guard i < chars.count, chars[i] == "{" else { return nil }
    i += 1
    var fields: [String: String] = [:]
    i = skipWhitespace(chars, i)
    if i < chars.count, chars[i] == "}" {
        return skipWhitespace(chars, i + 1) == chars.count ? fields : nil
    }
    while true {
        i = skipWhitespace(chars, i)
        guard i < chars.count, chars[i] == "\"", let keyEnd = scanValue(chars, i),
              let key = decodeJsonString(String(chars[i..<keyEnd]))
        else { return nil }
        i = skipWhitespace(chars, keyEnd)
        guard i < chars.count, chars[i] == ":" else { return nil }
        i = skipWhitespace(chars, i + 1)
        guard let valueEnd = scanValue(chars, i) else { return nil }
        fields[key] = String(chars[i..<valueEnd])
        i = skipWhitespace(chars, valueEnd)
        guard i < chars.count else { return nil }
        if chars[i] == "," {
            i += 1
        } else if chars[i] == "}" {
            return skipWhitespace(chars, i + 1) == chars.count ? fields : nil
        } else {
            return nil
        }
    }
}

/// Decode a JSON string literal, or `nil` if `raw` is not one.
///
/// Decoding runs over UTF-16 units so a `\u` surrogate pair rebuilds into the
/// one character it encodes rather than two unpaired halves.
func decodeJsonString(_ raw: String) -> String? {
    let chars = Array(raw)
    guard chars.count >= 2, chars[0] == "\"", chars[chars.count - 1] == "\"" else { return nil }
    var units: [UInt16] = []
    var i = 1
    let end = chars.count - 1
    while i < end {
        let ch = chars[i]
        guard ch == "\\" else {
            units.append(contentsOf: String(ch).utf16)
            i += 1
            continue
        }
        i += 1
        guard i < end else { return nil }
        switch chars[i] {
        case "\"", "\\", "/": units.append(contentsOf: String(chars[i]).utf16)
        case "b": units.append(0x08)
        case "f": units.append(0x0C)
        case "n": units.append(0x0A)
        case "r": units.append(0x0D)
        case "t": units.append(0x09)
        case "u":
            guard i + 5 <= end, let value = UInt16(String(chars[(i + 1)...(i + 4)]), radix: 16)
            else { return nil }
            units.append(value)
            i += 4
        default: return nil
        }
        i += 1
    }
    return String(decoding: units, as: UTF16.self)
}

/// Index just past the JSON value starting at `start`, or `nil` if it is
/// malformed or runs off the end (which is what a truncated reply looks like).
private func scanValue(_ chars: [Character], _ start: Int) -> Int? {
    guard start < chars.count else { return nil }
    switch chars[start] {
    case "\"":
        var i = start + 1
        while i < chars.count {
            if chars[i] == "\\" {
                i += 2
            } else if chars[i] == "\"" {
                return i + 1
            } else {
                i += 1
            }
        }
        return nil
    case "{", "[":
        var depth = 0
        var i = start
        while i < chars.count {
            switch chars[i] {
            case "{", "[":
                depth += 1
            case "}", "]":
                depth -= 1
                if depth == 0 { return i + 1 }
            case "\"":
                guard let end = scanValue(chars, i) else { return nil }
                i = end - 1
            default:
                break
            }
            i += 1
        }
        return nil
    default:
        var i = start
        while i < chars.count, !",}] \t\n\r".contains(chars[i]) { i += 1 }
        return i == start ? nil : i
    }
}

private func skipWhitespace(_ chars: [Character], _ start: Int) -> Int {
    var i = start
    while i < chars.count, chars[i].isWhitespace { i += 1 }
    return i
}

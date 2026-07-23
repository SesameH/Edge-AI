// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

public struct ChatMessage: Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }

    public static func user(_ content: String) -> ChatMessage { ChatMessage(role: "user", content: content) }
    public static func assistant(_ content: String) -> ChatMessage { ChatMessage(role: "assistant", content: content) }
    public static func system(_ content: String) -> ChatMessage { ChatMessage(role: "system", content: content) }
}

/// Memory footprint of a loaded model. Byte fields are `-1` when the
/// backend cannot measure them (distinct from a real zero).
public struct LlmRuntimeStats: Sendable {
    public let modelBytes: Int64
    public let kvCacheBytes: Int64
    public let devicePeakBytes: Int64
    public let processRssBytes: Int64
    public let deviceName: String?
}

/// Sampling controls; the defaults mean greedy decoding. At most one of
/// `grammar` / `jsonMode` / `jsonSchema` may be set — they all constrain
/// decoding through the same grammar slot.
public struct GenerateOptions: Sendable {
    public var maxTokens: Int32
    public var temperature: Float
    public var topP: Float
    public var topK: Int32
    public var seed: Int32
    /// Inline GBNF grammar constraining the output (optional).
    public var grammar: String?
    /// Constrain the output to syntactically valid JSON.
    public var jsonMode: Bool
    /// JSON Schema (serialized JSON text) the output must conform to.
    public var jsonSchema: String?

    public init(
        maxTokens: Int32 = 512,
        temperature: Float = 0,
        topP: Float = 0,
        topK: Int32 = 0,
        seed: Int32 = 0,
        grammar: String? = nil,
        jsonMode: Bool = false,
        jsonSchema: String? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.seed = seed
        self.grammar = grammar
        self.jsonMode = jsonMode
        self.jsonSchema = jsonSchema
    }
}

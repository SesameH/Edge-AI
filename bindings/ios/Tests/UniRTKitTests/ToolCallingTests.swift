// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import XCTest

@testable import UniRTKit

/// Pure-logic cover for tool calling — no model, no native handle, so these
/// run everywhere. The end-to-end counterpart (does the schema actually
/// constrain llama.cpp?) lives in InferenceSmokeTests.
final class ToolCallingTests: XCTestCase {
    private let weather = ToolDefinition(
        name: "get_weather",
        description: "Look up the weather",
        parametersJson: #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
    )
    private let ping = ToolDefinition(name: "ping")

    func testAutoSchemaOffersOneBranchPerToolPlusText() throws {
        let plan = try XCTUnwrap(ToolPlan.make(tools: [weather, ping]))

        // Byte-for-byte what the Kotlin binding emits for the same tools, and
        // structurally what the Python binding emits — the three bindings are
        // meant to constrain a model identically. The caller's parameter
        // schema appears verbatim, property order intact, and the no-argument
        // tool still gets a value the grammar can accept.
        XCTAssertEqual(
            plan.schemaJson,
            #"{"oneOf":["#
                + #"{"type":"object","properties":{"name":{"const":"get_weather"},"#
                + #""arguments":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},"#
                + #""required":["name","arguments"],"additionalProperties":false},"#
                + #"{"type":"object","properties":{"name":{"const":"ping"},"#
                + #""arguments":{"type":"object","properties":{},"additionalProperties":false}},"#
                + #""required":["name","arguments"],"additionalProperties":false},"#
                + #"{"type":"object","properties":{"content":{"type":"string"}},"#
                + #""required":["content"],"additionalProperties":false}]}"#
        )
    }

    func testRequiredDropsTheTextBranch() throws {
        let plan = try XCTUnwrap(ToolPlan.make(tools: [weather, ping], choice: .required))

        XCTAssertTrue(plan.schemaJson.contains(#""const":"get_weather""#))
        XCTAssertFalse(plan.schemaJson.contains(#""content""#))
        XCTAssertFalse(plan.systemPrompt.contains("To answer the user directly"))
    }

    func testNamingOneToolCollapsesToThatSingleBranch() throws {
        let plan = try XCTUnwrap(ToolPlan.make(tools: [weather, ping], choice: .function(name: "ping")))

        // One branch means no oneOf wrapper at all.
        XCTAssertTrue(plan.schemaJson.hasPrefix(#"{"type":"object""#))
        XCTAssertFalse(plan.schemaJson.contains("get_weather"))
    }

    func testNoneMeansNoPlan() throws {
        XCTAssertNil(try ToolPlan.make(tools: [weather], choice: .none))
    }

    func testTheSystemPromptDescribesEveryToolAndBothReplyShapes() throws {
        let plan = try XCTUnwrap(ToolPlan.make(tools: [weather, ping]))

        XCTAssertEqual(
            plan.systemPrompt,
            """
            You have access to these tools:

            - get_weather: Look up the weather
              parameters: {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
            - ping: no description provided
              parameters: {"type":"object","properties":{},"additionalProperties":false}

            Reply with a single JSON object and nothing else. To call a tool, use {"name": "<tool name>", "arguments": {...}}.
            To answer the user directly instead, use {"content": "<your reply>"}.
            """
        )
    }

    func testRejectsEmptyDuplicateAndUndeclaredTools() {
        XCTAssertThrowsError(try ToolPlan.make(tools: [])) {
            XCTAssertEqual($0 as? ToolCallingError, .noTools)
        }
        XCTAssertThrowsError(try ToolPlan.make(tools: [ping, ping])) {
            XCTAssertEqual($0 as? ToolCallingError, .duplicateToolName("ping"))
        }
        XCTAssertThrowsError(try ToolPlan.make(tools: [ToolDefinition(name: "")])) {
            XCTAssertEqual($0 as? ToolCallingError, .invalidToolName(""))
        }
        XCTAssertThrowsError(try ToolPlan.make(tools: [ping], choice: .function(name: "nope"))) {
            XCTAssertEqual($0 as? ToolCallingError, .undeclaredTool("nope"))
        }
    }

    func testACallIsReturnedWithItsArgumentsVerbatim() throws {
        let plan = try XCTUnwrap(ToolPlan.make(tools: [weather]))

        guard case .call(let call) = plan.interpret(
            #"{"name": "get_weather", "arguments": {"city": "Taipei"}}"#
        ) else { return XCTFail("expected a tool call") }

        XCTAssertEqual(call.name, "get_weather")
        XCTAssertEqual(call.argumentsJson, #"{"city": "Taipei"}"#)
        XCTAssertTrue(call.id.hasPrefix("call_"))
    }

    func testTheTextBranchIsUnescapedBackToPlainText() throws {
        let plan = try XCTUnwrap(ToolPlan.make(tools: [weather]))

        guard case .text(let text) = plan.interpret(
            #"{"content": "line\none \"quoted\" é 😀"}"#
        ) else { return XCTFail("expected text") }

        XCTAssertEqual(text, "line\none \"quoted\" é 😀")
    }

    func testAnEscapedSurrogatePairDecodesToTheCharacterItEncodes() throws {
        let plan = try XCTUnwrap(ToolPlan.make(tools: [weather]))

        // Assembled from the backslash so the source really carries the two
        // six-character escapes a model would emit, rather than the emoji
        // itself — that is the case the UTF-16 decoding path exists for.
        let backslash = "\u{5C}"
        let pair = backslash + "ud83d" + backslash + "ude00"

        guard case .text(let text) = plan.interpret(#"{"content": "a \#(pair) b"}"#) else {
            return XCTFail("expected text")
        }
        XCTAssertEqual(text, "a 😀 b")
    }

    func testATruncatedReplySurfacesAsTextRatherThanAFabricatedCall() throws {
        let plan = try XCTUnwrap(ToolPlan.make(tools: [weather]))

        // What maxTokens cutting a call mid-arguments looks like.
        let cut = #"{"name": "get_weather", "arguments": {"city": "Tai"#
        guard case .text(let text) = plan.interpret(cut) else {
            return XCTFail("a truncated call must not become a call")
        }
        XCTAssertEqual(text, cut)
    }

    func testAnUndeclaredToolNameIsNotTrusted() throws {
        let plan = try XCTUnwrap(ToolPlan.make(tools: [weather]))

        let text = #"{"name": "rm_rf", "arguments": {}}"#
        guard case .text(let passthrough) = plan.interpret(text) else {
            return XCTFail("an undeclared tool name must not become a call")
        }
        XCTAssertEqual(passthrough, text)
    }

    func testHistoryTurnsReplayACallAsTheJsonTheGrammarProduced() {
        let call = ToolCall(id: "call_1", name: "get_weather", argumentsJson: #"{"city": "Taipei"}"#)

        XCTAssertEqual(
            ChatMessage.toolCall(call).content,
            #"{"name": "get_weather", "arguments": {"city": "Taipei"}}"#
        )
        XCTAssertEqual(ChatMessage.toolCall(call).role, "assistant")
        XCTAssertEqual(
            ChatMessage.toolResult(name: "get_weather", content: "21C").content,
            "Result from get_weather: 21C"
        )
        XCTAssertEqual(ChatMessage.toolResult(name: nil, content: "21C").content, "Tool result: 21C")
        XCTAssertEqual(ChatMessage.toolResult(name: nil, content: "21C").role, "user")
    }
}

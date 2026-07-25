// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import CUniRT
import XCTest

@testable import UniRTKit

/// Exercises the real llama_cpp static plugin end to end — load, chat
/// template, greedy generate — the Swift-layer counterpart to
/// tests/native/test_inference_smoke.cpp. Needs the static libraries linked
/// in (see README.md) and UNIRT_TEST_MODEL_PATH pointing at a GGUF model;
/// skips otherwise.
final class InferenceSmokeTests: XCTestCase {
    func testGreedyGenerateProducesText() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["UNIRT_TEST_MODEL_PATH"],
              !modelPath.isEmpty
        else {
            throw XCTSkip("UNIRT_TEST_MODEL_PATH not set; skipping inference smoke test")
        }

        try UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)
        try UniRT.start()
        defer { try? UniRT.stop() }

        XCTAssertTrue(UniRT.plugins.contains("llama_cpp"))

        let session = try await UniRT.createLlmSession(modelPath: modelPath, nCtx: 256, nGpuLayers: 0)
        let prompt = try await session.applyChatTemplate([.user("Say hello in one word.")])
        XCTAssertFalse(prompt.isEmpty)

        let reply = try await session.generate(prompt: prompt, options: GenerateOptions(maxTokens: 16))
        XCTAssertFalse(reply.isEmpty)

        var pieces: [String] = []
        for try await piece in session.stream(prompt: prompt, options: GenerateOptions(maxTokens: 16)) {
            pieces.append(piece)
        }
        XCTAssertFalse(pieces.joined().isEmpty)

        try await session.reset()
        await session.close()
    }

    /// The tool-calling logic is covered without a model in ToolCallingTests;
    /// what needs a real backend is the wiring — that the generated schema
    /// actually reaches the sampler. A 135M model is far too small to pick the
    /// right tool, but `required` makes that irrelevant: the grammar cannot
    /// emit anything but one of these branches, so a reply that parses into a
    /// declared call proves the constraint was applied.
    func testRequiredToolChoiceConstrainsARealBackend() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["UNIRT_TEST_MODEL_PATH"],
              !modelPath.isEmpty
        else {
            throw XCTSkip("UNIRT_TEST_MODEL_PATH not set; skipping inference smoke test")
        }

        try UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)
        try UniRT.start()
        defer { try? UniRT.stop() }

        let session = try await UniRT.createLlmSession(modelPath: modelPath, nCtx: 512, nGpuLayers: 0)
        defer { Task { await session.close() } }

        let reply = try await session.chatWithTools(
            [.user("What is the weather in Taipei?")],
            tools: [
                ToolDefinition(
                    name: "get_weather",
                    description: "Look up the weather",
                    parametersJson: #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
                )
            ],
            toolChoice: .required,
            options: GenerateOptions(maxTokens: 64)
        )

        guard case .call(let call) = reply else {
            return XCTFail("required tool choice must produce a call, got \(reply)")
        }
        XCTAssertEqual(call.name, "get_weather")
        XCTAssertTrue(call.argumentsJson.contains("city"), call.argumentsJson)
    }
}

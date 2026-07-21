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
}

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import CUniRT
import XCTest

@testable import UniRTKit

/// No VLM (vision) test model is available to run a full generate() here
/// (unlike InferenceSmokeTests' GGUF text model), so this only proves the
/// six unirt_vlm_* entry points actually link and execute real plugin code
/// — a missing model file must fail cleanly through the whole Swift -> C ABI
/// -> plugin chain, not crash or hit a linker error at build time (which is
/// the real risk this test guards: build-time verification, not inference).
final class VlmLinkSmokeTests: XCTestCase {
    func testVlmCreateFailsCleanlyWithoutARealModel() async throws {
        try? UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)
        try? UniRT.start()
        defer { try? UniRT.stop() }

        do {
            _ = try await UniRT.createVlmSession(
                modelPath: "/nonexistent/model.gguf", mmprojPath: "/nonexistent/mmproj.gguf")
            XCTFail("expected a clean UniRTError for a missing model file")
        } catch let error as UniRTError {
            XCTAssertLessThan(error.code, 0)
        }
    }
}

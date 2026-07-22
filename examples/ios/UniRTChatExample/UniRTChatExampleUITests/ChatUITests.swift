// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import XCTest

final class ChatUITests: XCTestCase {
    func testSendProducesAssistantReply() throws {
        let app = XCUIApplication()
        app.launch()

        let status = app.staticTexts.firstMatch
        XCTAssertTrue(status.waitForExistence(timeout: 30))
        let deadline = Date().addingTimeInterval(30)
        while !status.label.hasPrefix("ready"), Date() < deadline {
            Thread.sleep(forTimeInterval: 0.5)
        }
        XCTAssertTrue(status.label.hasPrefix("ready"), "model never became ready: \(status.label)")

        let input = app.textFields["Ask something..."]
        XCTAssertTrue(input.waitForExistence(timeout: 5))
        input.tap()
        input.typeText("What is the capital of France?")

        app.buttons["Send"].tap()

        // Poll for a non-empty assistant bubble to appear.
        let replyDeadline = Date().addingTimeInterval(60)
        var replyText = ""
        while replyText.isEmpty, Date() < replyDeadline {
            Thread.sleep(forTimeInterval: 0.5)
            let texts = app.staticTexts.allElementsBoundByIndex.map(\.label)
            if let candidate = texts.first(where: { $0.localizedCaseInsensitiveContains("paris") }) {
                replyText = candidate
            }
        }
        XCTAssertFalse(replyText.isEmpty, "no reply mentioning Paris appeared in time")
    }
}

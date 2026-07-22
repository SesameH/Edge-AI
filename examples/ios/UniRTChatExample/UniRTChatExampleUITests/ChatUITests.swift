// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import XCTest

final class ChatUITests: XCTestCase {
    private func waitForReady(_ app: XCUIApplication, timeout: TimeInterval = 30) -> XCUIElement {
        let status = app.staticTexts["statusText"]
        XCTAssertTrue(status.waitForExistence(timeout: timeout))
        let deadline = Date().addingTimeInterval(timeout)
        while !status.label.hasPrefix("ready"), Date() < deadline {
            Thread.sleep(forTimeInterval: 0.5)
        }
        XCTAssertTrue(status.label.hasPrefix("ready"), "model never became ready: \(status.label)")
        return status
    }

    private func typeSlowly(_ text: String, into field: XCUIElement) {
        field.tap()
        for ch in text {
            field.typeText(String(ch))
            Thread.sleep(forTimeInterval: 0.09)
        }
        Thread.sleep(forTimeInterval: 0.6)
    }

    private func waitForReply(_ app: XCUIApplication, containing keywords: [String], timeout: TimeInterval) -> String {
        let deadline = Date().addingTimeInterval(timeout)
        var matched = ""
        while matched.isEmpty, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.5)
            let texts = app.staticTexts.allElementsBoundByIndex.map(\.label)
            if let candidate = texts.first(where: { text in
                keywords.contains { text.localizedCaseInsensitiveContains($0) }
            }) {
                matched = candidate
            }
        }
        return matched
    }

    func testTextModeProducesAssistantReply() throws {
        let app = XCUIApplication()
        app.launch()
        _ = waitForReady(app)

        let deviceValue = app.staticTexts["deviceStatValue"]
        if deviceValue.waitForExistence(timeout: 5) {
            print("UNIRT device (text): \(deviceValue.label)")
        }

        let input = app.textFields["Ask something..."]
        XCTAssertTrue(input.waitForExistence(timeout: 5))
        typeSlowly("What is the capital of France?", into: input)

        app.buttons["Send"].tap()

        let reply = waitForReply(app, containing: ["paris"], timeout: 60)
        XCTAssertFalse(reply.isEmpty, "no reply mentioning Paris appeared in time")
    }

    func testVisionModeDescribesAttachedImage() throws {
        let app = XCUIApplication()
        app.launch()
        _ = waitForReady(app)

        app.segmentedControls["modePicker"].buttons["Vision"].tap()
        _ = waitForReady(app)

        let deviceValue = app.staticTexts["deviceStatValue"]
        if deviceValue.waitForExistence(timeout: 5) {
            print("UNIRT device (vision): \(deviceValue.label)")
        }

        Thread.sleep(forTimeInterval: 0.4)
        app.buttons["attachButton"].tap()
        Thread.sleep(forTimeInterval: 0.6)

        let input = app.textFields["Ask something..."]
        XCTAssertTrue(input.waitForExistence(timeout: 5))
        typeSlowly("What do you see in this image?", into: input)

        app.buttons["Send"].tap()

        // test-photo.jpg is a synthetic scene: red house, green field,
        // mountains, yellow sun.
        let reply = waitForReply(app, containing: ["house", "mountain", "sun", "field", "green", "sky"], timeout: 90)
        XCTAssertFalse(reply.isEmpty, "no reply describing the test image appeared in time")
    }
}

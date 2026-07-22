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
            Thread.sleep(forTimeInterval: 0.05)
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

    /// One continuous session — Text mode then, without relaunching the app,
    /// switch the segmented picker to Vision mode. Keeps the whole demo (and
    /// any screen recording of it) as a single unbroken app session instead
    /// of visibly leaving and re-entering the app between modes.
    func testTextThenVisionInOneSession() throws {
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

        let textReply = waitForReply(app, containing: ["paris"], timeout: 60)
        XCTAssertFalse(textReply.isEmpty, "no reply mentioning Paris appeared in time")
        Thread.sleep(forTimeInterval: 2.5) // hold the finished reply on screen before switching modes

        let modelNameText = app.staticTexts["modelNameText"]
        let visionSegment = app.segmentedControls["modePicker"].buttons["Vision"]
        // The segmented-control tap has occasionally not registered (mode
        // stays on Text); retry rather than fail once on a missed tap.
        var switchedToVision = false
        for _ in 0..<3 {
            visionSegment.tap()
            let deadline = Date().addingTimeInterval(10)
            while !modelNameText.label.contains("vlm-model"), Date() < deadline {
                Thread.sleep(forTimeInterval: 0.3)
            }
            if modelNameText.label.contains("vlm-model") {
                switchedToVision = true
                break
            }
        }
        XCTAssertTrue(switchedToVision, "mode picker never switched to Vision (model name: \(modelNameText.label))")

        _ = waitForReady(app)

        if deviceValue.waitForExistence(timeout: 5) {
            print("UNIRT device (vision): \(deviceValue.label)")
        }

        let attachButton = app.buttons["attachButton"]
        XCTAssertTrue(attachButton.waitForExistence(timeout: 5))
        attachButton.tap()
        Thread.sleep(forTimeInterval: 0.6)

        typeSlowly("What do you see in this image?", into: input)

        app.buttons["Send"].tap()

        // test-photo.jpg is a synthetic scene: red house, green field,
        // mountains, yellow sun.
        let visionReply = waitForReply(app, containing: ["house", "mountain", "sun", "field", "green", "sky"], timeout: 90)
        XCTAssertFalse(visionReply.isEmpty, "no reply describing the test image appeared in time")
        Thread.sleep(forTimeInterval: 3) // hold the finished reply on screen before the test (and app) exits
    }
}

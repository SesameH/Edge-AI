// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt.example.chat

import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Android counterpart of the iOS example's ChatUITests: one continuous
 * session — Text mode answers the capital-of-France question, then the
 * mode switch to Vision attaches the bundled test photo and asks about it.
 * Real inference end to end; generous timeouts because an emulator decodes
 * on interpreted-speed CPU features.
 */
@RunWith(AndroidJUnit4::class)
class ChatUiTest {
    @get:Rule
    val compose = createAndroidComposeRule<MainActivity>()

    private fun waitForReady(timeoutMs: Long = 120_000) {
        compose.waitUntil(timeoutMs) {
            compose.onAllNodes(hasTestTag("statusText")).fetchSemanticsNodes().isNotEmpty()
        }
        compose.waitUntil(timeoutMs) {
            compose.onAllNodes(hasTestTag("statusText") and hasText("ready", substring = true))
                .fetchSemanticsNodes().isNotEmpty()
        }
    }

    private fun waitForReply(keywords: List<String>, timeoutMs: Long) {
        compose.waitUntil(timeoutMs) {
            keywords.any { keyword ->
                compose.onAllNodes(hasText(keyword, substring = true, ignoreCase = true))
                    .fetchSemanticsNodes().isNotEmpty()
            }
        }
    }

    @Test
    fun textThenVisionInOneSession() {
        waitForReady()

        compose.onNodeWithTag("inputField").performTextInput("What is the capital of France?")
        compose.onNodeWithTag("sendButton").performClick()
        waitForReply(listOf("Paris"), timeoutMs = 300_000)
        // "Paris" can appear mid-stream while isBusy still disables the mode
        // switch — wait for generation to finish before tapping Vision.
        waitForReady(timeoutMs = 300_000)

        compose.onNodeWithTag("modeVision").performClick()
        compose.waitUntil(120_000) {
            compose.onAllNodes(hasTestTag("modelNameText") and hasText("vlm-model", substring = true))
                .fetchSemanticsNodes().isNotEmpty()
        }
        waitForReady(timeoutMs = 300_000)

        compose.onNodeWithTag("attachButton").performClick()
        compose.onNodeWithTag("inputField").performTextInput("What do you see in this image?")
        compose.onNodeWithTag("sendButton").performClick()

        // test-photo.jpg is a synthetic scene: red house, green field,
        // mountains, yellow sun.
        waitForReply(
            listOf("house", "mountain", "sun", "field", "green", "sky"),
            timeoutMs = 600_000,
        )
    }
}

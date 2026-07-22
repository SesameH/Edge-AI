// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

// Same visual language as the iOS example (dark, cyan/violet, monospaced,
// glass panels) so the two demo GIFs read as one product.

package ai.unirt.example.chat

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.safeDrawingPadding
import ai.unirt.GenerationProfile

private object Theme {
    val bg = Color(0.02f, 0.03f, 0.05f)
    val cyan = Color(0.14f, 0.90f, 0.94f)
    val violet = Color(0.62f, 0.48f, 1.0f)
    val dim = Color.White.copy(alpha = 0.45f)
    val panel = Color.White.copy(alpha = 0.06f)
    val stroke = Color.White.copy(alpha = 0.08f)
    val radius = 12.dp
}

private val mono = FontFamily.Monospace

@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    LaunchedEffect(Unit) { viewModel.start() }

    Box(modifier = Modifier.fillMaxSize().background(Theme.bg)) {
        Column(
            modifier = Modifier.fillMaxSize().safeDrawingPadding().imePadding().padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Header(viewModel)
            StatsRow(viewModel.profile)
            MessageLog(viewModel, modifier = Modifier.weight(1f))
            viewModel.attachedImagePath?.let { AttachmentChip(it) { viewModel.clearAttachment() } }
            Composer(viewModel)
            Text(
                viewModel.status,
                fontFamily = mono,
                fontSize = 10.sp,
                color = Theme.dim,
                modifier = Modifier.fillMaxWidth().testTag("statusText"),
            )
        }
    }
}

@Composable
private fun Header(viewModel: ChatViewModel) {
    Column(
        modifier = Modifier.fillMaxWidth().panel().padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Box(
                modifier = Modifier.size(8.dp).clip(CircleShape)
                    .background(if (viewModel.isBusy) Theme.violet else Theme.cyan)
            )
            Text("UNIRT", fontFamily = mono, fontWeight = FontWeight.Bold, color = Theme.cyan)
            Text(
                "— ${viewModel.modelName}",
                fontFamily = mono,
                fontSize = 11.sp,
                color = Theme.dim,
                maxLines = 1,
                modifier = Modifier.testTag("modelNameText"),
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(Theme.radius)).background(Color.White.copy(alpha = 0.04f)),
        ) {
            ChatMode.entries.forEach { mode ->
                val selected = viewModel.mode == mode
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .padding(3.dp)
                        .clip(RoundedCornerShape(9.dp))
                        .background(if (selected) Theme.cyan.copy(alpha = 0.18f) else Color.Transparent)
                        .clickable(enabled = !viewModel.isBusy) { viewModel.switchMode(mode) }
                        .padding(vertical = 8.dp)
                        .testTag("mode${mode.label}"),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        mode.label,
                        fontFamily = mono,
                        fontSize = 13.sp,
                        fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                        color = if (selected) Theme.cyan else Theme.dim,
                    )
                }
            }
        }
    }
}

@Composable
private fun StatsRow(profile: GenerationProfile?) {
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        item { StatCard("TTFT", profile?.let { "${it.ttft} ms" } ?: "—") }
        item { StatCard("PREFILL", profile?.let { "%.1f tok/s".format(it.prefillSpeed) } ?: "—") }
        item { StatCard("DECODE", profile?.let { "%.1f tok/s".format(it.decodeSpeed) } ?: "—", valueTag = "decodeStatValue") }
        item { StatCard("TOKENS", profile?.let { "${it.promptTokens}+${it.generatedTokens}" } ?: "—") }
        item { StatCard("STOP", profile?.stopReason ?: "—") }
    }
}

@Composable
private fun StatCard(label: String, value: String, valueTag: String = "") {
    Column(
        modifier = Modifier.panel().padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Text(label, fontFamily = mono, fontSize = 9.sp, color = Theme.dim)
        Text(
            value,
            fontFamily = mono,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold,
            color = Theme.cyan,
            modifier = if (valueTag.isEmpty()) Modifier else Modifier.testTag(valueTag),
        )
    }
}

@Composable
private fun MessageLog(viewModel: ChatViewModel, modifier: Modifier = Modifier) {
    val listState = rememberLazyListState()
    LaunchedEffect(viewModel.messages.lastOrNull()?.text) {
        if (viewModel.messages.isNotEmpty()) listState.scrollToItem(viewModel.messages.size - 1)
    }
    LazyColumn(
        state = listState,
        modifier = modifier.fillMaxWidth().panel().padding(10.dp).testTag("messageLog"),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        itemsIndexed(viewModel.messages) { _, message -> Bubble(message) }
    }
}

@Composable
private fun Bubble(message: DisplayMessage) {
    val isUser = message.role == "user"
    val accent = if (isUser) Theme.cyan else Theme.violet
    Row(modifier = Modifier.fillMaxWidth()) {
        if (isUser) Spacer(modifier = Modifier.width(32.dp).weight(1f))
        Column(
            modifier = Modifier
                .widthIn(max = 340.dp)
                .clip(RoundedCornerShape(Theme.radius))
                .background(accent.copy(alpha = 0.10f))
                .border(1.dp, accent.copy(alpha = 0.4f), RoundedCornerShape(Theme.radius))
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            message.imagePath?.let { path ->
                val bitmap = remember(path) { BitmapFactory.decodeFile(path) }
                bitmap?.let {
                    Image(
                        bitmap = it.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.size(width = 140.dp, height = 100.dp).clip(RoundedCornerShape(8.dp)),
                    )
                }
            }
            Text(message.text, fontFamily = mono, fontSize = 14.sp, color = Color.White.copy(alpha = 0.92f))
        }
        if (!isUser) Spacer(modifier = Modifier.width(32.dp).weight(1f))
    }
}

@Composable
private fun AttachmentChip(path: String, onRemove: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().panel().padding(8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        val bitmap = remember(path) { BitmapFactory.decodeFile(path) }
        bitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = null,
                contentScale = ContentScale.Crop,
                modifier = Modifier.size(44.dp).clip(RoundedCornerShape(8.dp))
                    .border(1.dp, Theme.cyan.copy(alpha = 0.5f), RoundedCornerShape(8.dp)),
            )
        }
        Text("test-photo.jpg", fontFamily = mono, fontSize = 11.sp, color = Theme.dim)
        Spacer(modifier = Modifier.weight(1f))
        Text(
            "Remove",
            fontFamily = mono,
            fontSize = 11.sp,
            color = Theme.violet,
            modifier = Modifier.clickable { onRemove() },
        )
    }
}

@Composable
private fun Composer(viewModel: ChatViewModel) {
    Row(
        modifier = Modifier.fillMaxWidth().panel().padding(10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        if (viewModel.mode == ChatMode.Vision) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(Theme.radius))
                    .background(Color.White.copy(alpha = 0.04f))
                    .clickable { viewModel.attachTestImage() }
                    .testTag("attachButton"),
                contentAlignment = Alignment.Center,
            ) {
                Text("+", fontFamily = mono, fontWeight = FontWeight.Bold, fontSize = 18.sp, color = Theme.cyan)
            }
        }

        Box(
            modifier = Modifier
                .weight(1f)
                .clip(RoundedCornerShape(Theme.radius))
                .background(Color.White.copy(alpha = 0.04f))
                .border(1.dp, Theme.stroke, RoundedCornerShape(Theme.radius))
                .padding(10.dp),
        ) {
            if (viewModel.input.isEmpty()) {
                Text("Ask something...", fontFamily = mono, fontSize = 14.sp, color = Theme.dim)
            }
            BasicTextField(
                value = viewModel.input,
                onValueChange = { viewModel.input = it },
                textStyle = TextStyle(fontFamily = mono, fontSize = 14.sp, color = Color.White),
                cursorBrush = SolidColor(Theme.cyan),
                modifier = Modifier.fillMaxWidth().testTag("inputField"),
            )
        }

        val sendEnabled = !viewModel.isBusy && viewModel.input.isNotEmpty()
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(Theme.radius))
                .background(if (sendEnabled) Theme.cyan else Theme.cyan.copy(alpha = 0.3f))
                .clickable(enabled = sendEnabled) { viewModel.send() }
                .padding(horizontal = 16.dp, vertical = 10.dp)
                .testTag("sendButton"),
        ) {
            Text("Send", fontFamily = mono, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Theme.bg)
        }
    }
}

private fun Modifier.panel(): Modifier =
    this.clip(RoundedCornerShape(Theme.radius))
        .background(Theme.panel)
        .border(1.dp, Theme.stroke, RoundedCornerShape(Theme.radius))

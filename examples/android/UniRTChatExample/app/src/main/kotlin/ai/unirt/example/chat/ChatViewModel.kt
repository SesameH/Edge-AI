// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt.example.chat

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import ai.unirt.ChatMessage
import ai.unirt.ContentPart
import ai.unirt.GenerateOptions
import ai.unirt.GenerationProfile
import ai.unirt.LlmSession
import ai.unirt.LlmStreamResult
import ai.unirt.RuntimeStats
import ai.unirt.UniRT
import ai.unirt.VlmChatMessage
import ai.unirt.VlmGenerateOptions
import ai.unirt.VlmSession
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class ChatMode(val label: String) { Text("Text"), Vision("Vision") }

data class DisplayMessage(val role: String, val text: String, val imagePath: String? = null)

/**
 * Mirrors the iOS example's ChatViewModel: two sessions (LLM + VLM), both
 * kept resident once loaded so switching modes never re-pays model load or
 * its memory spike; per-mode transcript and display history; streaming
 * replies appended token-by-token.
 *
 * Models ship in the APK's assets (see README) but the native layer wants
 * filesystem paths, so each asset is copied to filesDir once on first use.
 */
class ChatViewModel(application: Application) : AndroidViewModel(application) {
    var mode by mutableStateOf(ChatMode.Text)
        private set
    val messages = mutableStateListOf<DisplayMessage>()
    var input by mutableStateOf("")
    var status by mutableStateOf("loading model...")
        private set
    var isBusy by mutableStateOf(false)
        private set
    var modelName by mutableStateOf("")
        private set
    var profile by mutableStateOf<GenerationProfile?>(null)
        private set
    var stats by mutableStateOf<RuntimeStats?>(null)
        private set
    var attachedImagePath by mutableStateOf<String?>(null)
        private set

    private var llmSession: LlmSession? = null
    private var vlmSession: VlmSession? = null
    private val llmHistory = mutableListOf<ChatMessage>()
    private val vlmHistory = mutableListOf<VlmChatMessage>()
    private var llmMessages = listOf<DisplayMessage>()
    private var vlmMessages = listOf<DisplayMessage>()
    private var started = false
    private var statsJob: Job? = null

    val testImagePath: String? get() = ensureAsset("test-photo.jpg")

    fun start() {
        viewModelScope.launch { switchMode(ChatMode.Text) }
    }

    fun switchMode(newMode: ChatMode) {
        if (isBusy) return
        mode = newMode
        attachedImagePath = null

        llmSession?.takeIf { newMode == ChatMode.Text }?.let {
            messages.setAll(llmMessages)
            status = readyStatus()
            return
        }
        vlmSession?.takeIf { newMode == ChatMode.Vision }?.let {
            messages.setAll(vlmMessages)
            status = readyStatus()
            return
        }

        messages.clear()
        profile = null
        status = "loading model..."
        isBusy = true

        viewModelScope.launch {
            try {
                if (!started) {
                    UniRT.start()
                    started = true
                }
                when (newMode) {
                    ChatMode.Text -> {
                        val modelPath = withContext(Dispatchers.IO) { ensureAsset("model.gguf") }
                        if (modelPath == null) {
                            status = "model.gguf not bundled — see examples/android/UniRTChatExample/README.md"
                            return@launch
                        }
                        modelName = File(modelPath).name
                        llmSession = UniRT.createLlmSession(modelPath, nCtx = 2048)
                    }
                    ChatMode.Vision -> {
                        val modelPath = withContext(Dispatchers.IO) { ensureAsset("vlm-model.gguf") }
                        if (modelPath == null) {
                            status = "vlm-model.gguf not bundled — see examples/android/UniRTChatExample/README.md"
                            return@launch
                        }
                        modelName = File(modelPath).name
                        val mmprojPath = withContext(Dispatchers.IO) { ensureAsset("mmproj.gguf") }
                        vlmSession = UniRT.createVlmSession(modelPath, mmprojPath = mmprojPath, nCtx = 2048)
                    }
                }
                status = readyStatus()
                startStatsPolling()
            } catch (e: Exception) {
                status = "load failed: ${e.message}"
            } finally {
                isBusy = false
            }
        }
    }

    /** Live memory/device stats every 2 s while idle — generation owns the
     *  session's single native thread, so polling waits it out via isBusy. */
    private fun startStatsPolling() {
        statsJob?.cancel()
        statsJob = viewModelScope.launch {
            while (isActive) {
                if (!isBusy) {
                    stats = try {
                        when (mode) {
                            ChatMode.Text -> llmSession?.runtimeStats()
                            ChatMode.Vision -> vlmSession?.runtimeStats()
                        }
                    } catch (_: Exception) {
                        null
                    }
                }
                delay(2_000)
            }
        }
    }

    fun attachTestImage() {
        attachedImagePath = testImagePath
    }

    fun clearAttachment() {
        attachedImagePath = null
    }

    fun send() {
        val text = input.trim()
        if (text.isEmpty() || isBusy) return
        input = ""
        isBusy = true
        status = "generating..."
        when (mode) {
            ChatMode.Text -> sendText(text)
            ChatMode.Vision -> sendVision(text)
        }
    }

    private fun sendText(text: String) {
        val session = llmSession ?: run { isBusy = false; return }
        llmHistory += ChatMessage.user(text)
        messages += DisplayMessage("user", text)
        val replyIndex = messages.size
        messages += DisplayMessage("assistant", "")

        viewModelScope.launch {
            try {
                val prompt = session.applyChatTemplate(llmHistory)
                var full = ""
                session.stream(prompt, GenerateOptions(maxTokens = 128)).collect { event ->
                    when (event) {
                        is LlmStreamResult.Token -> {
                            full += event.text
                            messages[replyIndex] = messages[replyIndex].copy(text = full)
                        }
                        is LlmStreamResult.Completed -> profile = event.profile
                        is LlmStreamResult.Error -> throw event.cause
                    }
                }
                llmHistory += ChatMessage.assistant(full)
                status = readyStatus()
            } catch (e: Exception) {
                messages[replyIndex] = messages[replyIndex].copy(text = "error: ${e.message}")
                status = readyStatus()
            }
            isBusy = false
            llmMessages = messages.toList()
        }
    }

    private fun sendVision(text: String) {
        val session = vlmSession ?: run { isBusy = false; return }
        val imagePath = attachedImagePath
        attachedImagePath = null

        val contents = if (imagePath != null) {
            listOf(ContentPart.Text(text), ContentPart.Image(imagePath))
        } else {
            listOf(ContentPart.Text(text))
        }
        vlmHistory += VlmChatMessage("user", contents)
        messages += DisplayMessage("user", text, imagePath)
        val replyIndex = messages.size
        messages += DisplayMessage("assistant", "")

        viewModelScope.launch {
            try {
                val prompt = session.applyChatTemplate(vlmHistory)
                // The template re-renders the whole transcript, so every image
                // marker in it — past turns included — needs its file supplied
                // again, in order; sending only the new image fails with
                // "prompt has N media markers but M files were supplied".
                // The prefix cache keeps re-supplied old images cheap.
                val allImages = vlmHistory.flatMap { it.contents }
                    .filterIsInstance<ContentPart.Image>()
                    .map { it.path }
                val options = VlmGenerateOptions(maxTokens = 160, imagePaths = allImages)
                var full = ""
                session.stream(prompt, options).collect { event ->
                    when (event) {
                        is LlmStreamResult.Token -> {
                            full += event.text
                            messages[replyIndex] = messages[replyIndex].copy(text = full)
                        }
                        is LlmStreamResult.Completed -> profile = event.profile
                        is LlmStreamResult.Error -> throw event.cause
                    }
                }
                vlmHistory += VlmChatMessage.assistant(full)
                status = readyStatus()
            } catch (e: Exception) {
                messages[replyIndex] = messages[replyIndex].copy(text = "error: ${e.message}")
                status = readyStatus()
            }
            isBusy = false
            vlmMessages = messages.toList()
        }
    }

    private fun readyStatus() = "ready (${UniRT.plugins().joinToString(", ")})"

    /** Copy an APK asset to filesDir once; null when the asset isn't bundled. */
    private fun ensureAsset(name: String): String? {
        val context = getApplication<Application>()
        val target = File(context.filesDir, name)
        return try {
            context.assets.open(name).use { input ->
                val size = input.available().toLong()
                if (!target.exists() || target.length() != size) {
                    target.outputStream().use { input.copyTo(it) }
                }
            }
            target.absolutePath
        } catch (_: java.io.FileNotFoundException) {
            null
        }
    }

    private fun <T> MutableList<T>.setAll(items: List<T>) {
        clear()
        addAll(items)
    }
}

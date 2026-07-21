// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

package ai.unirt

/** Receives streamed token pieces; return false to stop generation. */
fun interface TokenCallback {
    fun onToken(piece: String): Boolean
}

/** Raw JNI surface. Application code uses [UniRT] and [LlmSession]. */
internal object Native {
    init {
        System.loadLibrary("unirt_jni")
    }

    @JvmStatic external fun init(): Int
    @JvmStatic external fun deinit(): Int
    @JvmStatic external fun version(): String
    @JvmStatic external fun lastError(): String
    @JvmStatic external fun errorMessage(code: Int): String
    @JvmStatic external fun pluginList(): Array<String>

    @JvmStatic external fun llmCreate(
        modelPath: String, pluginId: String, deviceId: String?, nCtx: Int, nGpuLayers: Int,
    ): Long
    @JvmStatic external fun llmDestroy(handle: Long): Int
    @JvmStatic external fun llmReset(handle: Long): Int
    @JvmStatic external fun llmApplyChatTemplate(
        handle: Long, roles: Array<String>, contents: Array<String>, addGenerationPrompt: Boolean,
    ): String?
    @JvmStatic external fun llmGenerate(
        handle: Long, prompt: String, maxTokens: Int, temperature: Float, topP: Float,
        topK: Int, seed: Int, onToken: TokenCallback?,
    ): LlmGenerateResult?

    @JvmStatic external fun vlmCreate(
        modelPath: String, mmprojPath: String?, pluginId: String, deviceId: String?,
        nCtx: Int, nGpuLayers: Int,
    ): Long
    @JvmStatic external fun vlmDestroy(handle: Long): Int
    @JvmStatic external fun vlmReset(handle: Long): Int
    @JvmStatic external fun vlmGetCapabilities(handle: Long): VlmCapabilities?
    @JvmStatic external fun vlmApplyChatTemplate(
        handle: Long,
        roles: Array<String>,
        contentTypes: Array<Array<String>>,
        contentTexts: Array<Array<String>>,
        enableThinking: Boolean,
        grounding: Boolean,
    ): String?
    @JvmStatic external fun vlmGenerate(
        handle: Long, prompt: String, maxTokens: Int, temperature: Float, topP: Float,
        topK: Int, seed: Int, imagePaths: Array<String>, audioPaths: Array<String>,
        imageMaxLength: Int, onToken: TokenCallback?,
    ): LlmGenerateResult?
}

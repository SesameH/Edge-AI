// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import CUniRT

/// One piece of a multimodal turn: plain text, or a path to an image/audio file.
public enum ContentPart: Sendable {
    case text(String)
    case image(path: String)
    case audio(path: String)

    var wireType: String {
        switch self {
        case .text: return "text"
        case .image: return "image"
        case .audio: return "audio"
        }
    }

    var wireText: String {
        switch self {
        case .text(let text): return text
        case .image(let path): return path
        case .audio(let path): return path
        }
    }
}

public struct VlmChatMessage: Sendable {
    public let role: String
    public let contents: [ContentPart]

    public init(role: String, contents: [ContentPart]) {
        self.role = role
        self.contents = contents
    }

    public static func user(_ contents: ContentPart...) -> VlmChatMessage { VlmChatMessage(role: "user", contents: contents) }
    public static func user(_ text: String) -> VlmChatMessage { VlmChatMessage(role: "user", contents: [.text(text)]) }
    public static func assistant(_ text: String) -> VlmChatMessage { VlmChatMessage(role: "assistant", contents: [.text(text)]) }
    public static func system(_ text: String) -> VlmChatMessage { VlmChatMessage(role: "system", contents: [.text(text)]) }
}

/// Which media the loaded projector actually accepts (see `VlmSession.capabilities()`).
public struct VlmCapabilities: Sendable {
    public let supportsVision: Bool
    public let supportsAudio: Bool
}

/// Sampling controls plus per-request media. Kept separate from `GenerateOptions`
/// rather than adding image/audio fields there: those fields would be dead
/// weight on every LLM call, which never consumes them.
public struct VlmGenerateOptions: Sendable {
    public var maxTokens: Int32
    public var temperature: Float
    public var topP: Float
    public var topK: Int32
    public var seed: Int32
    public var imagePaths: [String]
    public var audioPaths: [String]
    /// Cap on the longest image edge; 0 = no resize.
    public var imageMaxLength: Int32

    public init(
        maxTokens: Int32 = 512,
        temperature: Float = 0,
        topP: Float = 0,
        topK: Int32 = 0,
        seed: Int32 = 0,
        imagePaths: [String] = [],
        audioPaths: [String] = [],
        imageMaxLength: Int32 = 0
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.seed = seed
        self.imagePaths = imagePaths
        self.audioPaths = audioPaths
        self.imageMaxLength = imageMaxLength
    }
}

/// One loaded multimodal model. Obtain from `UniRT.createVlmSession`. Same
/// threading contract as `LlmSession` — an actor confining every native call.
public actor VlmSession {
    private var handle: OpaquePointer?

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    static func open(
        modelPath: String, mmprojPath: String?, pluginId: String, deviceId: String?,
        nCtx: Int32, nGpuLayers: Int32
    ) async throws -> VlmSession {
        var config = unirt_ModelConfig()
        config.n_ctx = nCtx
        config.n_gpu_layers = nGpuLayers

        var out: OpaquePointer?
        let status = modelPath.withCString { modelPathPtr in
            pluginId.withCString { pluginIdPtr in
                withOptionalCString(mmprojPath) { mmprojPtr in
                    withOptionalCString(deviceId) { deviceIdPtr in
                        var input = unirt_VlmCreateInput()
                        input.model_path = modelPathPtr
                        input.mmproj_path = mmprojPtr
                        input.plugin_id = pluginIdPtr
                        input.device_id = deviceIdPtr
                        input.config = config
                        return unirt_vlm_create(&input, &out)
                    }
                }
            }
        }
        try UniRTError.check(status)
        guard let handle = out else {
            throw UniRTError(code: -1, detail: "unirt_vlm_create returned no handle")
        }
        return VlmSession(handle: handle)
    }

    /// Which media the loaded projector accepts; llama_cpp reflects the
    /// mmproj, other plugins may report both false.
    public func capabilities() throws -> VlmCapabilities {
        var caps = unirt_VlmCapabilities()
        try UniRTError.check(unirt_vlm_get_capabilities(try requireOpen(), &caps))
        return VlmCapabilities(supportsVision: caps.supports_vision, supportsAudio: caps.supports_audio)
    }

    /// Render a multimodal conversation through the model's chat template.
    public func applyChatTemplate(
        _ messages: [VlmChatMessage], enableThinking: Bool = false, grounding: Bool = false
    ) throws -> String {
        let handle = try requireOpen()
        return try withVlmMessages(messages) { nativeMessages -> String in
            var messages = nativeMessages
            var input = unirt_VlmApplyChatTemplateInput()
            var output = unirt_VlmApplyChatTemplateOutput()
            let status = messages.withUnsafeMutableBufferPointer { buffer -> Int32 in
                input.messages = buffer.baseAddress
                input.message_count = Int32(buffer.count)
                input.enable_thinking = enableThinking
                input.grounding = grounding
                return unirt_vlm_apply_chat_template(handle, &input, &output)
            }
            try UniRTError.check(status)
            defer { unirt_free(output.formatted_text) }
            return output.formatted_text.map { String(cString: $0) } ?? ""
        }
    }

    /// Generate to completion and return the full reply.
    public func generate(prompt: String, options: VlmGenerateOptions = VlmGenerateOptions()) throws -> String {
        try runGenerate(prompt: prompt, options: options, onToken: nil)
    }

    /// Generate as a cold stream of token pieces; cancelling the task stops decoding.
    public nonisolated func stream(
        prompt: String, options: VlmGenerateOptions = VlmGenerateOptions()
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    _ = try await self.runGenerate(prompt: prompt, options: options) { piece in
                        switch continuation.yield(piece) {
                        case .terminated: return false
                        default: return true
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Template + generate in one call.
    public func chat(_ messages: [VlmChatMessage], options: VlmGenerateOptions = VlmGenerateOptions()) throws -> String {
        try generate(prompt: applyChatTemplate(messages), options: options)
    }

    /// Drop the conversation state (KV cache and sampler state).
    public func reset() throws {
        try UniRTError.check(unirt_vlm_reset(try requireOpen()))
    }

    /// Unload the model. Safe to call more than once.
    public func close() {
        guard let handle else { return }
        unirt_vlm_destroy(handle)
        self.handle = nil
    }

    deinit {
        if let handle { unirt_vlm_destroy(handle) }
    }

    private func requireOpen() throws -> OpaquePointer {
        guard let handle else { throw UniRTError(code: -1, detail: "session is closed") }
        return handle
    }

    private func runGenerate(
        prompt: String, options: VlmGenerateOptions, onToken: ((String) -> Bool)?
    ) throws -> String {
        let handle = try requireOpen()

        var sampler = unirt_SamplerConfig()
        sampler.temperature = options.temperature
        sampler.top_p = options.topP
        sampler.top_k = options.topK
        sampler.seed = options.seed

        var config = unirt_GenerationConfig()
        config.max_tokens = options.maxTokens
        config.image_max_length = options.imageMaxLength

        return try prompt.withCString { promptPtr -> String in
            try withCStringArray(options.imagePaths) { imagePtrs -> String in
                try withCStringArray(options.audioPaths) { audioPtrs -> String in
                    try withUnsafeMutablePointer(to: &sampler) { samplerPtr -> String in
                        config.sampler_config = samplerPtr
                        return try imagePtrs.withUnsafeBufferPointer { imageBuffer -> String in
                            config.image_paths = imageBuffer.isEmpty ? nil : UnsafeMutablePointer(mutating: imageBuffer.baseAddress)
                            config.image_count = Int32(imageBuffer.count)
                            return try audioPtrs.withUnsafeBufferPointer { audioBuffer -> String in
                                config.audio_paths = audioBuffer.isEmpty ? nil : UnsafeMutablePointer(mutating: audioBuffer.baseAddress)
                                config.audio_count = Int32(audioBuffer.count)
                                return try withUnsafeMutablePointer(to: &config) { configPtr -> String in
                                    var input = unirt_VlmGenerateInput()
                                    input.prompt_utf8 = promptPtr
                                    input.config = UnsafePointer(configPtr)

                                    let relay = onToken.map(TokenRelay.init)
                                    let relayBox = relay.map { Unmanaged.passRetained($0) }
                                    defer { relayBox?.release() }
                                    if relayBox != nil {
                                        input.on_token = unirt_token_trampoline
                                        input.user_data = relayBox!.toOpaque()
                                    }

                                    var output = unirt_VlmGenerateOutput()
                                    let status = unirt_vlm_generate(handle, &input, &output)
                                    try UniRTError.check(status)
                                    defer { unirt_free(output.full_text) }
                                    return output.full_text.map { String(cString: $0) } ?? ""
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Runs `body` with a native `unirt_VlmChatMessage` array whose `contents`
/// pointers (and everything they reference) stay valid for the call —
/// two-level version of `withCStringArray`: each message needs its own
/// owned array of (type, text) pairs alive at the same time as the outer
/// per-message struct array.
private func withVlmMessages<R>(
    _ messages: [VlmChatMessage], _ body: ([unirt_VlmChatMessage]) throws -> R
) rethrows -> R {
    func recurse(_ index: Int, _ acc: inout [unirt_VlmChatMessage]) throws -> R {
        if index == messages.count { return try body(acc) }
        let message = messages[index]
        return try message.role.withCString { rolePtr in
            let types = message.contents.map(\.wireType)
            let texts = message.contents.map(\.wireText)
            return try withCStringArray(types) { typePtrs in
                try withCStringArray(texts) { textPtrs in
                    var contents = zip(typePtrs, textPtrs).map { unirt_VlmContent(type: $0, text: $1) }
                    return try contents.withUnsafeMutableBufferPointer { buffer -> R in
                        acc.append(unirt_VlmChatMessage(
                            role: rolePtr, contents: buffer.baseAddress, content_count: Int64(buffer.count)))
                        defer { acc.removeLast() }
                        return try recurse(index + 1, &acc)
                    }
                }
            }
        }
    }
    var acc: [unirt_VlmChatMessage] = []
    return try recurse(0, &acc)
}

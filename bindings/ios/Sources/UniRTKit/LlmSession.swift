// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import CUniRT

/// Boxes a Swift closure behind an opaque pointer so it can ride as
/// `user_data` through the C callback and back.
final class TokenRelay {
    let onToken: (String) -> Bool
    init(onToken: @escaping (String) -> Bool) { self.onToken = onToken }
}

func unirt_token_trampoline(
    _ token: UnsafePointer<CChar>?, _ userData: UnsafeMutableRawPointer?
) -> Bool {
    guard let userData else { return false }
    let relay = Unmanaged<TokenRelay>.fromOpaque(userData).takeUnretainedValue()
    return relay.onToken(token.map { String(cString: $0) } ?? "")
}

/// One loaded text model. Obtain from `UniRT.createLlmSession`. The native
/// handle is single-threaded by contract — every native call is confined to
/// this actor, matching that contract without extra locking. Call `close()`
/// (or let the session deinit) before `UniRT.stop()`.
public actor LlmSession {
    private var handle: OpaquePointer?

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    static func open(
        modelPath: String, pluginId: String, deviceId: String?, nCtx: Int32, nGpuLayers: Int32
    ) async throws -> LlmSession {
        var config = unirt_ModelConfig()
        config.n_ctx = nCtx
        config.n_gpu_layers = nGpuLayers

        var out: OpaquePointer?
        let status = modelPath.withCString { modelPathPtr in
            pluginId.withCString { pluginIdPtr in
                withOptionalCString(deviceId) { deviceIdPtr in
                    var input = unirt_LlmCreateInput()
                    input.model_path = modelPathPtr
                    input.plugin_id = pluginIdPtr
                    input.device_id = deviceIdPtr
                    input.config = config
                    return unirt_llm_create(&input, &out)
                }
            }
        }
        try UniRTError.check(status)
        guard let handle = out else {
            throw UniRTError(code: -1, detail: "unirt_llm_create returned no handle")
        }
        return LlmSession(handle: handle)
    }

    /// Render a conversation through the model's chat template.
    public func applyChatTemplate(_ messages: [ChatMessage], addGenerationPrompt: Bool = true) throws -> String {
        let handle = try requireOpen()
        let roles = messages.map(\.role)
        let contents = messages.map(\.content)
        return try withCStringArray(roles) { rolePtrs in
            try withCStringArray(contents) { contentPtrs in
                var nativeMessages = zip(rolePtrs, contentPtrs).map {
                    unirt_LlmChatMessage(role: $0, content: $1)
                }
                var input = unirt_LlmApplyChatTemplateInput()
                var output = unirt_LlmApplyChatTemplateOutput()
                let status = nativeMessages.withUnsafeMutableBufferPointer { buffer -> Int32 in
                    input.messages = buffer.baseAddress
                    input.message_count = Int32(buffer.count)
                    input.add_generation_prompt = addGenerationPrompt
                    return unirt_llm_apply_chat_template(handle, &input, &output)
                }
                try UniRTError.check(status)
                defer { unirt_free(output.formatted_text) }
                return output.formatted_text.map { String(cString: $0) } ?? ""
            }
        }
    }

    /// Generate to completion and return the full reply.
    public func generate(prompt: String, options: GenerateOptions = GenerateOptions()) throws -> String {
        try runGenerate(prompt: prompt, options: options, onToken: nil)
    }

    /// Generate as a cold stream of token pieces; cancelling the task stops
    /// decoding. Resending a growing transcript reuses the KV prefix.
    public nonisolated func stream(
        prompt: String, options: GenerateOptions = GenerateOptions()
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
    public func chat(_ messages: [ChatMessage], options: GenerateOptions = GenerateOptions()) throws -> String {
        try generate(prompt: applyChatTemplate(messages), options: options)
    }

    /// Drop the conversation state (KV cache and transcript).
    public func reset() throws {
        try UniRTError.check(unirt_llm_reset(try requireOpen()))
    }

    /// Memory footprint of the loaded model — weights, KV cache, device peak,
    /// process RSS, and the active compute device name. Cheap; safe to poll.
    public func runtimeStats() throws -> LlmRuntimeStats {
        let handle = try requireOpen()
        var output = unirt_LlmRuntimeStats()
        try UniRTError.check(unirt_llm_get_runtime_stats(handle, &output))
        return LlmRuntimeStats(
            modelBytes: output.model_bytes,
            kvCacheBytes: output.kv_cache_bytes,
            devicePeakBytes: output.device_peak_bytes,
            processRssBytes: output.process_rss_bytes,
            deviceName: output.device_name.map { String(cString: $0) }
        )
    }

    /// Unload the model. Safe to call more than once.
    public func close() {
        guard let handle else { return }
        unirt_llm_destroy(handle)
        self.handle = nil
    }

    deinit {
        if let handle { unirt_llm_destroy(handle) }
    }

    private func requireOpen() throws -> OpaquePointer {
        guard let handle else { throw UniRTError(code: -1, detail: "session is closed") }
        return handle
    }

    private func runGenerate(
        prompt: String, options: GenerateOptions, onToken: ((String) -> Bool)?
    ) throws -> String {
        let handle = try requireOpen()

        var sampler = unirt_SamplerConfig()
        sampler.temperature = options.temperature
        sampler.top_p = options.topP
        sampler.top_k = options.topK
        sampler.seed = options.seed
        sampler.enable_json = options.jsonMode

        var config = unirt_GenerationConfig()
        config.max_tokens = options.maxTokens

        return try withOptionalCString(options.grammar) { grammarPtr -> String in
        try withOptionalCString(options.jsonSchema) { schemaPtr -> String in
        sampler.grammar_string = grammarPtr
        sampler.json_schema = schemaPtr
        return try prompt.withCString { promptPtr -> String in
            try withUnsafeMutablePointer(to: &sampler) { samplerPtr -> String in
                config.sampler_config = samplerPtr
                return try withUnsafeMutablePointer(to: &config) { configPtr -> String in
                    var input = unirt_LlmGenerateInput()
                    input.prompt_utf8 = promptPtr
                    input.config = UnsafePointer(configPtr)

                    let relay = onToken.map(TokenRelay.init)
                    let relayBox = relay.map { Unmanaged.passRetained($0) }
                    defer { relayBox?.release() }
                    if relayBox != nil {
                        input.on_token = unirt_token_trampoline
                        input.user_data = relayBox!.toOpaque()
                    }

                    var output = unirt_LlmGenerateOutput()
                    let status = unirt_llm_generate(handle, &input, &output)
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

/// Runs `body` with `value` as a live C string pointer, or `nil` when
/// `value` is `nil` — `String.withCString` has no nil-friendly overload.
func withOptionalCString<R>(_ value: String?, _ body: (UnsafePointer<CChar>?) throws -> R) rethrows -> R {
    guard let value else { return try body(nil) }
    return try value.withCString(body)
}

/// Runs `body` with an array of live C string pointers, one per element,
/// all valid for the duration of the call.
func withCStringArray<R>(_ values: [String], _ body: ([UnsafePointer<CChar>?]) throws -> R) rethrows -> R {
    func recurse(_ index: Int, _ acc: inout [UnsafePointer<CChar>?]) throws -> R {
        if index == values.count { return try body(acc) }
        return try values[index].withCString { ptr in
            acc.append(ptr)
            defer { acc.removeLast() }
            return try recurse(index + 1, &acc)
        }
    }
    var acc: [UnsafePointer<CChar>?] = []
    return try recurse(0, &acc)
}

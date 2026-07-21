// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import CUniRT

/// Runtime lifecycle and session factory.
///
/// ```swift
/// UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)
/// try UniRT.start()
/// let session = try await UniRT.createLlmSession(modelPath: "/path/to/model.gguf")
/// for try await piece in session.stream(prompt: try await session.applyChatTemplate([.user("Hi")])) {
///     print(piece, terminator: "")
/// }
/// try UniRT.stop()
/// ```
public enum UniRT {
    /// Scan and load the bundled backend plugins. iOS forbids dlopen of
    /// arbitrary paths, so every backend must first join via
    /// `registerStaticPlugin` — call that before `start()`.
    public static func start() throws {
        try UniRTError.check(unirt_init())
    }

    /// Unload plugins; every session must be closed first.
    public static func stop() throws {
        try UniRTError.check(unirt_deinit())
    }

    public static var version: String { String(cString: unirt_version()) }

    public static var plugins: [String] {
        var output = unirt_GetPluginListOutput()
        guard unirt_get_plugin_list(&output) == UNIRT_SUCCESS.rawValue, output.plugin_count > 0 else { return [] }
        defer { unirt_free(output.plugin_ids) }
        return (0..<Int(output.plugin_count)).map { String(cString: output.plugin_ids![$0]!) }
    }

    /// Joins a backend linked directly into the binary (the loading model
    /// iOS requires: no dlopen of arbitrary paths). Pass the plugin's
    /// `unirt_plugin_id` / `unirt_plugin_open` C entry points, e.g. for the
    /// static llama_cpp plugin linked from `libunirt_llama_cpp.a`:
    /// `UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)`.
    /// Call before `start()`.
    public static func registerStaticPlugin(
        identity: @convention(c) () -> UnsafePointer<CChar>?,
        open: @convention(c) () -> OpaquePointer?
    ) throws {
        try UniRTError.check(unirt_register_plugin(identity, open))
    }

    /// Load a model and return its session. Heavy: on-disk models and
    /// multi-GB prefill mean this can take seconds, hence `async`.
    public static func createLlmSession(
        modelPath: String,
        pluginId: String = "llama_cpp",
        deviceId: String? = nil,
        nCtx: Int32 = 0,
        nGpuLayers: Int32 = -1
    ) async throws -> LlmSession {
        try await LlmSession.open(
            modelPath: modelPath, pluginId: pluginId, deviceId: deviceId,
            nCtx: nCtx, nGpuLayers: nGpuLayers)
    }

    /// Load a multimodal model and return its session.
    public static func createVlmSession(
        modelPath: String,
        mmprojPath: String? = nil,
        pluginId: String = "llama_cpp",
        deviceId: String? = nil,
        nCtx: Int32 = 0,
        nGpuLayers: Int32 = -1
    ) async throws -> VlmSession {
        try await VlmSession.open(
            modelPath: modelPath, mmprojPath: mmprojPath, pluginId: pluginId, deviceId: deviceId,
            nCtx: nCtx, nGpuLayers: nGpuLayers)
    }
}

// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import CUniRT
import Foundation
import UniRTKit

enum ChatMode: String, CaseIterable {
    case text = "Text"
    case vision = "Vision"
}

struct DisplayMessage: Identifiable {
    let id = UUID()
    let role: String
    var text: String
    var imagePath: String?
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var mode: ChatMode = .text
    @Published var messages: [DisplayMessage] = []
    @Published var input: String = ""
    @Published var status: String = "loading model..."
    @Published var isBusy: Bool = false
    @Published var modelName: String = ""
    @Published var stats: LlmRuntimeStats?
    @Published var attachedImagePath: String?

    private var llmSession: LlmSession?
    private var vlmSession: VlmSession?
    private var llmHistory: [ChatMessage] = []
    private var vlmHistory: [VlmChatMessage] = []
    private var statsTask: Task<Void, Never>?
    private var pluginRegistered = false

    var testImagePath: String? { Bundle.main.path(forResource: "test-photo", ofType: "jpg") }

    func start() async {
        await switchMode(to: .text)
    }

    func switchMode(to newMode: ChatMode) async {
        statsTask?.cancel()
        await llmSession?.close()
        await vlmSession?.close()
        llmSession = nil
        vlmSession = nil
        messages = []
        attachedImagePath = nil
        stats = nil
        mode = newMode
        status = "loading model..."

        do {
            if !pluginRegistered {
                try UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)
                try UniRT.start()
                pluginRegistered = true
            }
            switch newMode {
            case .text:
                guard let modelPath = Bundle.main.path(forResource: "model", ofType: "gguf") else {
                    status = "model.gguf not bundled — see examples/ios/UniRTChatExample/README.md"
                    return
                }
                modelName = (modelPath as NSString).lastPathComponent
                llmHistory = []
                llmSession = try await UniRT.createLlmSession(modelPath: modelPath, nCtx: 2048)
                status = "ready (\(UniRT.plugins.joined(separator: ", ")))"
            case .vision:
                guard let modelPath = Bundle.main.path(forResource: "vlm-model", ofType: "gguf") else {
                    status = "vlm-model.gguf not bundled — see examples/ios/UniRTChatExample/README.md"
                    return
                }
                let mmprojPath = Bundle.main.path(forResource: "mmproj", ofType: "gguf")
                modelName = (modelPath as NSString).lastPathComponent
                vlmHistory = []
                vlmSession = try await UniRT.createVlmSession(modelPath: modelPath, mmprojPath: mmprojPath, nCtx: 2048)
                let caps = try await vlmSession?.capabilities()
                status = "ready (\(UniRT.plugins.joined(separator: ", "))) — vision: \(caps?.supportsVision == true)"
            }
            startStatsPolling()
        } catch {
            status = "load failed: \(error)"
        }
    }

    private func startStatsPolling() {
        statsTask?.cancel()
        statsTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                if !self.isBusy {
                    if let session = self.llmSession {
                        self.stats = try? await session.runtimeStats()
                    } else if let session = self.vlmSession {
                        self.stats = try? await session.runtimeStats()
                    }
                }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    func attachTestImage() {
        attachedImagePath = testImagePath
    }

    func clearAttachment() {
        attachedImagePath = nil
    }

    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isBusy else { return }
        input = ""
        isBusy = true
        status = "generating..."

        switch mode {
        case .text:
            sendText(text)
        case .vision:
            sendVision(text)
        }
    }

    private func sendText(_ text: String) {
        guard let session = llmSession else { isBusy = false; return }
        llmHistory.append(.user(text))
        messages.append(DisplayMessage(role: "user", text: text))
        let replyIndex = messages.count
        messages.append(DisplayMessage(role: "assistant", text: ""))

        Task {
            do {
                let prompt = try await session.applyChatTemplate(llmHistory)
                var full = ""
                for try await piece in session.stream(prompt: prompt, options: GenerateOptions(maxTokens: 128)) {
                    full += piece
                    messages[replyIndex].text = full
                }
                llmHistory.append(.assistant(full))
                status = "ready"
            } catch {
                messages[replyIndex].text = "error: \(error)"
                status = "ready"
            }
            isBusy = false
        }
    }

    private func sendVision(_ text: String) {
        guard let session = vlmSession else { isBusy = false; return }
        let imagePath = attachedImagePath
        attachedImagePath = nil

        let contents: [ContentPart] = imagePath.map { [.text(text), .image(path: $0)] } ?? [.text(text)]
        vlmHistory.append(VlmChatMessage(role: "user", contents: contents))
        messages.append(DisplayMessage(role: "user", text: text, imagePath: imagePath))
        let replyIndex = messages.count
        messages.append(DisplayMessage(role: "assistant", text: ""))

        Task {
            do {
                let prompt = try await session.applyChatTemplate(vlmHistory)
                let options = VlmGenerateOptions(maxTokens: 160, imagePaths: imagePath.map { [$0] } ?? [])
                var full = ""
                for try await piece in session.stream(prompt: prompt, options: options) {
                    full += piece
                    messages[replyIndex].text = full
                }
                vlmHistory.append(.assistant(full))
                status = "ready"
            } catch {
                messages[replyIndex].text = "error: \(error)"
                status = "ready"
            }
            isBusy = false
        }
    }
}

func formatBytes(_ n: Int64?) -> String {
    guard let n, n >= 0 else { return "n/a" }
    if n == 0 { return "0 B" }
    let units = ["B", "KB", "MB", "GB"]
    var value = Double(n)
    var unitIndex = 0
    while value >= 1024, unitIndex < units.count - 1 {
        value /= 1024
        unitIndex += 1
    }
    let decimals = (value >= 10 || unitIndex == 0) ? 0 : 1
    return String(format: "%.\(decimals)f %@", value, units[unitIndex])
}

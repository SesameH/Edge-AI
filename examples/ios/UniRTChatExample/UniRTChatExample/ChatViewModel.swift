// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import CUniRT
import Foundation
import UniRTKit

struct DisplayMessage: Identifiable {
    let id = UUID()
    let role: String
    var text: String
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [DisplayMessage] = []
    @Published var input: String = ""
    @Published var status: String = "loading model..."
    @Published var isBusy: Bool = false
    @Published var modelName: String = ""
    @Published var stats: LlmRuntimeStats?

    private var session: LlmSession?
    private var history: [ChatMessage] = []
    private var statsTask: Task<Void, Never>?

    func start() async {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "gguf") else {
            status = "model.gguf not bundled — see examples/ios/README.md"
            return
        }
        modelName = (modelPath as NSString).lastPathComponent
        do {
            try UniRT.registerStaticPlugin(identity: unirt_plugin_id, open: unirt_plugin_open)
            try UniRT.start()
            session = try await UniRT.createLlmSession(modelPath: modelPath, nCtx: 2048)
            status = "ready (\(UniRT.plugins.joined(separator: ", ")))"
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
                if !self.isBusy, let session = self.session {
                    self.stats = try? await session.runtimeStats()
                }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let session, !isBusy else { return }
        input = ""
        history.append(.user(text))
        messages.append(DisplayMessage(role: "user", text: text))
        let replyIndex = messages.count
        messages.append(DisplayMessage(role: "assistant", text: ""))
        isBusy = true
        status = "generating..."

        Task {
            do {
                let prompt = try await session.applyChatTemplate(history)
                var full = ""
                for try await piece in session.stream(prompt: prompt, options: GenerateOptions(maxTokens: 128)) {
                    full += piece
                    messages[replyIndex].text = full
                }
                history.append(.assistant(full))
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

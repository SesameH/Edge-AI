// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import SwiftUI

private enum Theme {
    static let bg = Color(red: 0.02, green: 0.03, blue: 0.05)
    static let cyan = Color(red: 0.14, green: 0.90, blue: 0.94)
    static let violet = Color(red: 0.62, green: 0.48, blue: 1.0)
    static let dim = Color.white.opacity(0.45)
    static let radius: CGFloat = 12
}

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()

    var body: some View {
        ZStack {
            background

            VStack(spacing: 10) {
                header
                statsRow
                log
                composer
            }
            .padding(12)
        }
        .preferredColorScheme(.dark)
        .task { await viewModel.start() }
    }

    private var background: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            Circle()
                .fill(Theme.cyan.opacity(0.25))
                .frame(width: 340, height: 340)
                .blur(radius: 90)
                .offset(x: -140, y: -300)
            Circle()
                .fill(Theme.violet.opacity(0.20))
                .frame(width: 340, height: 340)
                .blur(radius: 90)
                .offset(x: 160, y: -260)
        }
        .ignoresSafeArea()
    }

    private var header: some View {
        HStack {
            HStack(spacing: 6) {
                Circle()
                    .fill(viewModel.isBusy ? Theme.violet : Theme.cyan)
                    .frame(width: 8, height: 8)
                Text("UNIRT")
                    .font(.system(.callout, design: .monospaced)).bold()
                    .foregroundStyle(Theme.cyan)
                Text("— \(viewModel.modelName)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(Theme.dim)
                    .lineLimit(1)
            }
            Spacer()
        }
        .padding(12)
        .glassPanel()
    }

    private var statsRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                statCard("DEVICE", viewModel.stats?.deviceName ?? "—")
                statCard("MODEL MEM", formatBytes(viewModel.stats?.modelBytes))
                statCard("KV CACHE", formatBytes(viewModel.stats?.kvCacheBytes))
                statCard("DEVICE PEAK", formatBytes(viewModel.stats?.devicePeakBytes))
                statCard("PROCESS RSS", formatBytes(viewModel.stats?.processRssBytes))
            }
        }
        .opacity(viewModel.isBusy ? 0.5 : 1.0)
        .animation(.easeInOut(duration: 0.2), value: viewModel.isBusy)
    }

    private func statCard(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(Theme.dim)
            Text(value)
                .font(.system(.footnote, design: .monospaced)).bold()
                .foregroundStyle(Theme.cyan)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .glassPanel()
    }

    private var log: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(viewModel.messages) { message in
                        bubble(for: message).id(message.id)
                    }
                }
                .padding(4)
            }
            .onChange(of: viewModel.messages.last?.text) { _ in
                if let lastId = viewModel.messages.last?.id {
                    withAnimation { proxy.scrollTo(lastId, anchor: .bottom) }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(10)
        .glassPanel()
    }

    private var composer: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                TextField("", text: $viewModel.input, prompt: Text("Ask something...").foregroundColor(Theme.dim))
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.white)
                    .padding(10)
                    .background(Color.white.opacity(0.04))
                    .clipShape(RoundedRectangle(cornerRadius: Theme.radius))
                    .overlay(RoundedRectangle(cornerRadius: Theme.radius).stroke(Color.white.opacity(0.08)))
                    .accessibilityLabel("Ask something...")
                    .onSubmit { viewModel.send() }

                Button("Send") { viewModel.send() }
                    .font(.system(.footnote, design: .monospaced)).bold()
                    .foregroundStyle(Theme.bg)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(viewModel.isBusy || viewModel.input.isEmpty ? Theme.cyan.opacity(0.3) : Theme.cyan)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.radius))
                    .disabled(viewModel.isBusy || viewModel.input.isEmpty)
            }
            .padding(10)
            .glassPanel()

            Text(viewModel.status)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Theme.dim)
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityIdentifier("statusText")
        }
    }

    @ViewBuilder
    private func bubble(for message: DisplayMessage) -> some View {
        let isUser = message.role == "user"
        HStack {
            if isUser { Spacer(minLength: 32) }
            Text(message.text)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.white.opacity(0.92))
                .padding(12)
                .background((isUser ? Theme.cyan : Theme.violet).opacity(0.10))
                .clipShape(RoundedRectangle(cornerRadius: Theme.radius))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.radius)
                        .stroke((isUser ? Theme.cyan : Theme.violet).opacity(0.4), lineWidth: 1)
                )
            if !isUser { Spacer(minLength: 32) }
        }
    }
}

private struct GlassPanel: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: Theme.radius))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.radius)
                    .stroke(Color.white.opacity(0.08), lineWidth: 1)
            )
    }
}

private extension View {
    func glassPanel() -> some View { modifier(GlassPanel()) }
}

#Preview {
    ContentView()
}

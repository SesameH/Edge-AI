// Copyright (c) 2026 Peter Huang.
// SPDX-License-Identifier: BSD-3-Clause

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()

    var body: some View {
        VStack(spacing: 0) {
            Text(viewModel.status)
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.vertical, 6)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(viewModel.messages) { message in
                            bubble(for: message).id(message.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.last?.text) { _ in
                    if let lastId = viewModel.messages.last?.id {
                        proxy.scrollTo(lastId, anchor: .bottom)
                    }
                }
            }

            HStack {
                TextField("Ask something...", text: $viewModel.input)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { viewModel.send() }
                Button("Send") { viewModel.send() }
                    .disabled(viewModel.isBusy || viewModel.input.isEmpty)
            }
            .padding()
        }
        .task { await viewModel.start() }
    }

    @ViewBuilder
    private func bubble(for message: DisplayMessage) -> some View {
        HStack {
            if message.role == "user" { Spacer(minLength: 40) }
            Text(message.text)
                .padding(10)
                .background(message.role == "user" ? Color.blue.opacity(0.15) : Color.gray.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 10))
            if message.role != "user" { Spacer(minLength: 40) }
        }
    }
}

#Preview {
    ContentView()
}

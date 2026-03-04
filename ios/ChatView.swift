import SwiftUI

struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel

    var body: some View {
        VStack(spacing: 12) {
            Text("Mindfulness AI")
                .font(.title2.weight(.semibold))
                .foregroundStyle(Color.cyan)

            Text("Share how you feel and receive a gentle, guided response.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(viewModel.messages) { msg in
                            HStack {
                                if msg.role == "assistant" {
                                    bubble(text: msg.content, isUser: false)
                                    Spacer(minLength: 40)
                                } else {
                                    Spacer(minLength: 40)
                                    bubble(text: msg.content, isUser: true)
                                }
                            }
                            .id(msg.id)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top, 4)
                }
                .onChange(of: viewModel.messages.count) { _ in
                    if let last = viewModel.messages.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            HStack(spacing: 8) {
                TextField("How are you feeling?", text: $viewModel.input)
                    .textFieldStyle(.roundedBorder)
                    .disabled(viewModel.isSending)

                Button {
                    Task { await viewModel.send() }
                } label: {
                    if viewModel.isSending {
                        ProgressView()
                    } else {
                        Text("Send")
                            .fontWeight(.semibold)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isSending)
            }
            .padding(.horizontal)
            .padding(.bottom, 12)
        }
    }

    @ViewBuilder
    private func bubble(text: String, isUser: Bool) -> some View {
        Text(text)
            .padding(10)
            .background(isUser ? Color.cyan.opacity(0.9) : Color(.systemGray5))
            .foregroundStyle(isUser ? Color.black : Color.primary)
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}


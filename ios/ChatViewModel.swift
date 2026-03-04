import Foundation

struct ChatMessage: Identifiable, Codable {
    let id = UUID()
    let role: String
    let content: String
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var input: String = ""
    @Published var isSending: Bool = false

    private let baseURL = ProcessInfo.processInfo.environment["API_BASE_URL"] ?? "http://localhost:8000"

    func send() async {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isSending else { return }

        isSending = true
        input = ""

        let userMessage = ChatMessage(role: "user", content: trimmed)
        messages.append(userMessage)

        do {
            let reply = try await sendToBackend(query: trimmed, history: messages)
            messages.append(reply)
        } catch {
            messages.append(ChatMessage(role: "assistant", content: "Sorry, something went wrong while contacting the server."))
        }

        isSending = false
    }

    private func sendToBackend(query: String, history: [ChatMessage]) async throws -> ChatMessage {
        guard let url = URL(string: "\(baseURL)/v1/mindfulness/session") else {
            throw URLError(.badURL)
        }

        struct RequestBody: Codable {
            let query: String
            let history: [ChatMessage]
        }

        struct ResponseBody: Codable {
            let message: String
            let transcript: String?
            let voice_character: String?
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = RequestBody(query: query, history: history)
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }

        let decoded = try JSONDecoder().decode(ResponseBody.self, from: data)
        return ChatMessage(role: "assistant", content: decoded.message)
    }
}


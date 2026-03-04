import SwiftUI

@main
struct MindfulnessApp: App {
    var body: some Scene {
        WindowGroup {
            ChatView(viewModel: ChatViewModel())
        }
    }
}


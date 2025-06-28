import SwiftUI

struct ContentView: View {
    // Use ObservedObject for the view model passed from the App
    @ObservedObject var viewModel: ChatViewModel
    @State private var userInput: String = ""
    @State private var isGenerating: Bool = false // State to track generation

    var body: some View {
        VStack(spacing: 0) {
            // Chat History Area
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(viewModel.messages) { message in
                            MessageView(message: message)
                                .id(message.id) // Assign ID for scrolling
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top, 10) // Add some padding at the top
                }
                .onChange(of: viewModel.messages) { _ , newMessages in
                    // Scroll to the bottom when new messages are added
                    if let lastMessage = newMessages.last {
                        withAnimation {
                            proxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                    }
                }
                .onAppear {
                    // Scroll to the bottom initially if there are messages
                     if let lastMessage = viewModel.messages.last {
                        proxy.scrollTo(lastMessage.id, anchor: .bottom)
                    }
                }
            }

            Divider()

            // Input Area
            HStack {
                TextField("Type your message...", text: $userInput, axis: .vertical)
                    .textFieldStyle(.plain) // Basic style
                    .lineLimit(1...5) // Allow multiline input up to 5 lines
                    .padding(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
                    .background(Color(uiColor: .systemGray6)) // Subtle background
                    .cornerRadius(18) // Rounded corners
                    .disabled(viewModel.isInitializing || isGenerating) // Disable when initializing OR generating

                Button {
                    sendMessage()
                } label: {
                    Image(systemName: "arrow.up.circle.fill") // Use SF Symbols
                        .font(.system(size: 28)) // Adjust icon size
                }
                .disabled(userInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isInitializing || isGenerating) // Disable if input empty OR initializing OR generating
                .padding(.leading, 5)
            }
            .padding() // Padding around the input HStack
            .background(.thinMaterial) // Material background for separation
        }
        .navigationTitle("Llama 1 Chat") // Basic title
        .navigationBarTitleDisplayMode(.inline)
        // Add a task modifier to handle generation state updates
        // Use the Bool value directly for the task id
        .task(id: viewModel.isGenerating) { // Closure takes no arguments
            // Update the local state when the viewModel's state changes
            self.isGenerating = viewModel.isGenerating // Read the current value
        }
        // *** ADDED: Task modifier to initialize the ViewModel ***
        .task { await viewModel.initialize() }
        // Use the binding $viewModel.showErrorAlert for isPresented
        .alert("Error", isPresented: $viewModel.showErrorAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(viewModel.errorMessage ?? "An unknown error occurred.")
        }
    }

    // Function to handle sending message
    private func sendMessage() {
        let messageText = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        if !messageText.isEmpty {
            // Immediately clear input and set generating state
            userInput = ""
            isGenerating = true // Visually disable input immediately

            // Call ViewModel to process and generate response
            Task {
                 await viewModel.sendMessage(messageText)
                 // ViewModel will set its isGenerating back to false when done
            }
        }
    }
}

// Simple View for displaying a single message
struct MessageView: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user {
                Spacer() // Push user messages to the right
                Text(message.content)
                    .padding(10)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(15)
                    .frame(maxWidth: .infinity, alignment: .trailing) // Align within the cell
            } else {
                Text(message.content)
                    .padding(10)
                    .background(Color(uiColor: .systemGray4))
                    .foregroundColor(.primary)
                    .cornerRadius(15)
                    .frame(maxWidth: .infinity, alignment: .leading) // Align within the cell
                Spacer() // Push bot messages to the left
            }
        }
         .frame(maxWidth: .infinity) // Ensure HStack takes full width for alignment
    }
}

// Preview Provider (Optional, update if needed)
#Preview {
    // Create and configure the ViewModel within the preview scope
    let previewViewModel: ChatViewModel = {
        let vm = ChatViewModel()
        vm.messages = [
            ChatMessage(role: .user, content: "Hello!"),
            ChatMessage(role: .assistant, content: "Hi there! How can I help you today?"),
            ChatMessage(role: .user, content: "Tell me about SwiftUI."),
            ChatMessage(role: .assistant, content: "SwiftUI is a declarative framework...")
        ]
        return vm
    }() // Immediately invoke the closure to get the configured ViewModel

    // Return the view directly as the final statement
    NavigationView { // Wrap in NavigationView for title
         ContentView(viewModel: previewViewModel)
    }
}

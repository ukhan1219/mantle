import Foundation
import Combine

@MainActor // Ensure UI updates happen on the main thread
class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var temperature: Float = 0.8 // Default from python script
    @Published var topP: Float = 0.9       // Default from python script
    @Published var maxTokens: Int = 100     // Default from python script
    @Published var isGenerating: Bool = false // Renamed from isLoading for clarity
    @Published var isInitializing: Bool = true
    @Published var errorMessage: String? = nil {
        didSet {
            // Automatically update showErrorAlert based on errorMessage
            showErrorAlert = (errorMessage != nil)
        }
    }
    @Published var showErrorAlert: Bool = false // Controls alert presentation

    // InferenceController will be initialized asynchronously
    private var inferenceController: InferenceController?

    // Default initial message or leave empty
    init() {
         // messages.append(ChatMessage(role: .assistant, content: "Hello! How can I help you today?"))
    }

    /// Asynchronous initializer function to be called from the View's .task modifier
    func initialize() async {
        if inferenceController == nil { // Prevent re-initialization
            print("ChatViewModel initializing...")
            isInitializing = true
            errorMessage = nil
            do {
                inferenceController = try await InferenceController()
                print("ChatViewModel initialization successful.")
            } catch let error as InferenceError {
                // Handle specific InferenceErrors
                print("ChatViewModel initialization failed (InferenceError): \(error)")
                switch error {
                case .modelLoadingFailed(let msg), .modelCompilationFailed(let msg), .tokenizerLoadFailed(let msg):
                    errorMessage = "Initialization Error: \(msg)"
                default:
                    errorMessage = "Initialization Error: An unexpected inference error occurred."
                }
            } catch {
                // Handle other unexpected errors
                print("ChatViewModel initialization failed (Unexpected Error): \(error)")
                errorMessage = "Failed to initialize inference engine: \(error.localizedDescription)"
            }
            isInitializing = false
            isGenerating = false // Ensure generating is false after init
        }
    }

    /// Sends the current prompt to the inference controller and handles the response stream.
    func sendMessage(_ text: String) async { // Now accepts text parameter
        guard let inferenceController = inferenceController else {
            errorMessage = "Inference engine not initialized."
            isGenerating = false // Ensure state is reset
            return
        }
        let promptToSend = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !promptToSend.isEmpty, !isGenerating else { return } // Check passed text and generating state

        // Set generating state immediately
        isGenerating = true
        errorMessage = nil

        // Append user message
        let userMessage = ChatMessage(role: .user, content: promptToSend)
        messages.append(userMessage)

        // Prepare placeholder for assistant response
        let assistantMessagePlaceholder = ChatMessage(role: .assistant, content: "")
        messages.append(assistantMessagePlaceholder)
        let assistantMessageIndex = messages.count - 1

        // No longer need to manage internal prompt property
        // self.prompt = ""

        // Removed nested Task, function is already async
        do {
            // Start generation
            let stream = try await inferenceController.generateResponse(
                prompt: promptToSend,
                maxTokens: maxTokens
            )

            // Handle the stream of generated tokens
            var streamedText = ""
            for try await token in stream {
                streamedText += token
                // Update the content of the last assistant message
                if messages.indices.contains(assistantMessageIndex) {
                    messages[assistantMessageIndex].content = streamedText.trimmingCharacters(in: .whitespacesAndNewlines) // Trim whitespace
                }
            }
            // Ensure final content is set if stream ends abruptly
            if messages.indices.contains(assistantMessageIndex) && messages[assistantMessageIndex].content.isEmpty {
                 messages[assistantMessageIndex].content = streamedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "(No response)" : streamedText.trimmingCharacters(in: .whitespacesAndNewlines)
            }

        } catch {
            // Handle errors during generation
            print("Error generating response: \(error)")
            let errorDesc = (error as? InferenceError)?.localizedDescription ?? error.localizedDescription
            errorMessage = "Error generating response: \(errorDesc)"
            // Update placeholder message to show error
            if messages.indices.contains(assistantMessageIndex) {
                 messages[assistantMessageIndex].content = "[Error generating response]"
            }
        }

        // Mark loading as finished regardless of success or failure
        isGenerating = false
    }

    // Function to clear the chat history
    func clearChat() {
        messages = []
        errorMessage = nil
        // Potentially add logic to reset inference controller state if needed
        // (May require re-initializing or adding a reset function to InferenceController)
    }
} 

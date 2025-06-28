import Foundation

/// Represents a single message in the chat conversation.
struct ChatMessage: Identifiable, Hashable {
    /// Unique identifier for the message.
    let id = UUID()
    /// The role of the entity that generated the message (user or assistant).
    var role: ChatRole
    /// The text content of the message.
    var content: String
}

/// Defines the possible roles in the chat.
enum ChatRole: String, Hashable {
    case user
    case assistant
} 
//
//  MantleApp.swift
//  Mantle
//
//  Created by Usman Khan on 4/28/25.
//

import SwiftUI

@main
struct MantleApp: App {
    // Create the ChatViewModel as a StateObject owned by the App
    @StateObject private var chatViewModel = ChatViewModel()

    var body: some Scene {
        WindowGroup {
            // Inject the viewModel into ContentView
            ContentView(viewModel: chatViewModel)
        }
    }
}

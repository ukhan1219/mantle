# Mantle: On-Device LLM Chat for iOS

[![Swift](https://img.shields.io/badge/Swift-5.0-orange.svg)](https://swift.org)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Core_ML](https://img.shields.io/badge/Core_ML-iOS_18-green.svg)](https://developer.apple.com/documentation/coreml)
[![Metal](https://img.shields.io/badge/Metal-GPU_Optimized-purple.svg)](https://developer.apple.com/metal/)

Mantle is an iOS-based chat application that runs powerful Large Language Models (LLMs) entirely on-device, leveraging the full potential of Apple Silicon. This project is a deep dive into the world of on-device AI, focusing on the significant engineering challenges involved in making large, complex models performant within the constrained environment of a mobile phone.

## Demo

*A screen-recorded demo of the app in action:*

[![Mantle App Demo](https://img.youtube.com/vi/J20uOGPSQi8/0.jpg)](https://www.youtube.com/watch?v=J20uOGPSQi8)

## 🚀 Project Purpose

The primary goal of this project is to explore and implement advanced optimization techniques for running LLMs on iOS devices. It serves as a portfolio piece to showcase deep technical expertise in:

*   **Model Compression & Quantization:** Making massive models fit and run on mobile.
*   **Stateful On-Device Inference:** Efficiently handling context and generation.
*   **Performance Profiling & Bottleneck Analysis:** Using tools like Xcode Instruments to scientifically measure and improve performance.
*   **Low-Level GPU Programming:** Implementing custom layers with Metal Performance Shaders (MPS) to accelerate model execution beyond what standard frameworks offer.

This project was born from a passion for AI development and a desire to push the boundaries of what's possible with on-device machine learning, demonstrating skills that are crucial for building the next generation of intelligent, private, and responsive mobile applications.

## ✨ How It Works: The Optimization Journey

Running a multi-billion parameter LLM on a phone is non-trivial. The memory footprint of the model weights and the computational cost of inference are significant hurdles. The key to making this work is a multi-stage optimization process.

### 1. Model Conversion & Compression

The first step is to convert a pre-trained model from a framework like PyTorch into Apple's Core ML format. This project currently uses `meta-llama/Llama-3.2-3B` as its primary model, with previous experiments on Mistral and DeepSeek models.

*   **Conversion:** Python scripts in `llama3-conversion/` use `coremltools` to perform the conversion.
*   **Quantization:** To shrink the model, we apply aggressive quantization, converting the model's weights from 16-bit floating-point numbers to smaller types like 8-bit integers. This drastically reduces the model's size and memory usage, a critical step for on-device deployment.

### 2. Stateful Inference & The KV Cache

For an LLM to have a conversation, it must remember the context of what's been said. This is managed by a **Key-Value (KV) Cache**. In a naive implementation, this cache is re-calculated at every step, which is incredibly slow.

This project implements **stateful inference**. The Core ML model is designed to accept the KV cache as an input `state`, and it outputs the updated cache after each prediction. This means the model only computes the *next* token, making generation significantly faster and more efficient. The `stateful_convert_llama3_to_coreml.py` script is specifically designed to create this stateful model.

### 3. Profiling & Identifying Bottlenecks

Once the model is running on the device, the next phase is to analyze its performance using **Xcode Instruments**. We measure:
*   **Latency:** Time to generate each token.
*   **Memory Usage:** Peak RAM consumption.
*   **Power Consumption:** Energy impact on the device.
*   **Compute Unit Utilization:** Which parts of the model run on the CPU, GPU, and the Apple Neural Engine (ANE).

This analysis reveals performance bottlenecks—typically the **Attention mechanism** layers, which are the computational heart of a Transformer model.

### 4. Metal Performance Shader (MPS) Optimization (Ongoing)

This is the most advanced and ongoing phase of the project. When profiling shows that a specific layer (e.g., an attention block) is a bottleneck, we can replace Core ML's default implementation for that layer with our own highly optimized version using **Metal**, Apple's low-level GPU programming framework.

The process involves:
1.  **Writing Custom Layers:** Implementing the mathematical operations of the bottleneck layer in Swift, using the **Metal Performance Shaders (MPS)** library for highly optimized functions like matrix multiplication.
2.  **Integration:** Using Core ML's `MLCustomLayer` protocol to seamlessly integrate our custom Metal code into the model's execution graph.
3.  **Re-profiling:** Measuring the performance uplift to validate the optimization.

This demonstrates the ability to go beyond standard frameworks and hand-tune performance for specific hardware, which is a crucial skill for high-performance mobile AI.

## 🛠️ Technology Stack

*   **Model Conversion:**
    *   Python, PyTorch, Transformers
    *   `coremltools` for conversion and quantization.
*   **iOS Application (Mantle):**
    *   Swift, SwiftUI for the user interface.
    *   `CoreML` for running the model.
    *   `Metal` and `Metal Performance Shaders (MPS)` for custom layer optimizations.
*   **Tooling:**
    *   Xcode & Xcode Instruments for development, profiling, and debugging.

## 📦 Project Components

*   **`llama3-conversion/`**: Contains Python scripts to download, convert, and quantize the Llama 3.2 model.
*   **`Mantle/`**: The Xcode project for the iOS SwiftUI application.
    *   `ContentView.swift`: The main chat UI.
    *   `ChatViewModel.swift`: Manages the state of the chat and interacts with the inference engine.
    *   `InferenceController.swift`: The core component that loads the Core ML model, manages the generation loop, and handles the stateful KV cache.
*   **`.mlpackage` file (Generated):** The final, compressed Core ML model that is bundled with the iOS app.

## 🔧 How to Replicate

1.  **Prerequisites:**
    *   macOS with Xcode 16 or newer.
    *   Python 3.9+ with a virtual environment.
    *   Access to the `meta-llama/Llama-3.2-3B` model on Hugging Face (requires an access token).

2.  **Set up Python Environment:**
    ```bash
    cd llama3-conversion
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Run Conversion Script:**
    *   Make sure you are logged into Hugging Face CLI (`huggingface-cli login`).
    *   Execute the conversion script. This will download the model, convert it, and save it as `Llama1Stateful_FP16.mlpackage` (and possibly a quantized version).
    ```bash
    python stateful_convert_llama3_to_coreml.py
    ```

4.  **Add Model to Xcode:**
    *   Locate the generated `.mlpackage` file in the `llama3-conversion` directory.
    *   Drag and drop the `.mlpackage` file into the `Mantle/Mantle/` folder within the Xcode project navigator. Ensure it is added to the "Mantle" target and included in "Copy Bundle Resources".

5.  **Build and Run:**
    *   Open `Mantle.xcodeproj` in Xcode.
    *   Select a physical iOS device (required for Metal/ANE performance).
    *   Build and run the application.

## 🏆 Accomplishments & Learnings

*   **Accomplished:**
    *   Successfully converted and deployed a multi-billion parameter LLM (`Llama-3.2-3B`) on an iPhone.
    *   Implemented a fully stateful inference pipeline for efficient token generation.
    *   Established a robust workflow for profiling and analyzing on-device model performance.
    *   Built a functional and responsive SwiftUI chat interface.
*   **Learned:**
    *   The critical importance of KV cache management for LLM performance.
    *   The trade-offs between model size, quantization level, and output quality.
    *   Advanced usage of `coremltools` for stateful models and quantization.
    *   The process of identifying and preparing for low-level optimization with Metal.

## 🔮 Future Work

*   Continue the implementation of custom Metal layers for the attention mechanism.
*   Experiment with more aggressive quantization techniques (e.g., 4-bit) and analyze the performance vs. accuracy trade-off.
*   Integrate other models to test the flexibility of the pipeline.
*   Enhance the chat UI with features like conversation history and context management. 
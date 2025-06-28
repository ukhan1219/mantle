import Foundation
import CoreML
import Tokenizers // From swift-transformers package
import Hub        // From swift-transformers
import Accelerate // For sampling helpers if needed

// Define potential errors
enum InferenceError: Error {
    case modelLoadingFailed(String)
    case modelCompilationFailed(String) // Added for compilation step
    case tokenizerLoadFailed(String)
    case inputPreparationFailed(String)
    case predictionFailed(String)
    case outputProcessingFailed(String)
    case invalidInput(String)
    case samplingFailed(String)
}

/// Handles loading the Core ML model and performing stateful text generation.
class InferenceController {
    private let model: MLModel
    private let tokenizer: Tokenizer
    private let eosTokenId: Int // End of Sequence token ID

    // Configuration constants (adjust if needed)
    private let modelBaseName = "Llama1Stateful_FP16" // <<< CORRECT MODEL NAME
    // private let hfRepoId = "meta-llama/Llama-3.2-3B" // <<< No longer needed, loading locally
    // private let tokenizerDirectoryName = "llama3_tokenizer_bundle" // <<< Not searching for directory anymore
    private let tokenizerFileName = "tokenizer" // <<< Name of the main tokenizer file
    private let defaultEOS = 128001 // Default Llama 3 EOS, will try to override from tokenizer

    // Asynchronous initializer
    init() async throws {
        var finalModelURL: URL?
        var loadLog = "" // For detailed logging if needed

        do {
            // --- STEP 1: Try finding the COMPILED model URL (.mlmodelc) ---
            loadLog += "Attempting to find compiled model: \(modelBaseName).mlmodelc\n"
            if let compiledURL = Bundle.main.url(forResource: modelBaseName, withExtension: "mlmodelc") {
                loadLog += "  -> SUCCESS: Found pre-compiled URL: \(compiledURL.path)\n"
                finalModelURL = compiledURL
            } else {
                loadLog += "  -> FAILED: Pre-compiled model not found directly.\n"

                // --- STEP 2 (Fallback): Try finding the SOURCE model URL (.mlpackage) ---
                loadLog += "Attempting fallback to source model: \(modelBaseName).mlpackage\n"
                guard let sourceURL = Bundle.main.url(forResource: modelBaseName, withExtension: "mlpackage") else {
                    loadLog += "  -> FAILED: Source model (.mlpackage) not found either.\n"
                    print(loadLog) // Log the failure reason
                    throw InferenceError.modelLoadingFailed("Could not find model as .mlmodelc or .mlpackage named '\(modelBaseName)' in the app bundle.")
                }

                loadLog += "  -> SUCCESS: Found source URL: \(sourceURL.path)\n"
                // If source found, try compiling it at runtime
                do {
                    loadLog += "  Attempting runtime compilation...\n"
                    // Use MLModel.compileModel(at:)
                    let compiledRuntimeURL = try await MLModel.compileModel(at: sourceURL)
                    loadLog += "  Runtime compilation successful: \(compiledRuntimeURL.path)\n"
                    finalModelURL = compiledRuntimeURL // Use the newly compiled URL
                } catch {
                    loadLog += "  Runtime compilation FAILED: \(error.localizedDescription)\n"
                    if let nsError = error as NSError? { loadLog += "  Error details: \(nsError.userInfo)\n" }
                    print(loadLog) // Log the failure reason
                    throw InferenceError.modelCompilationFailed("Failed to compile model '\(modelBaseName).mlpackage': \(error.localizedDescription)")
                }
            }

            // --- Ensure we have a URL to load ---
            guard let modelToLoadURL = finalModelURL else {
                 // This case should technically be unreachable due to prior checks/throws, but safeguard anyway.
                 loadLog += "Critical Error: No usable model URL was determined despite checks.\n"
                 print(loadLog)
                 throw InferenceError.modelLoadingFailed("Internal error: Could not determine a valid model URL.")
            }

            // --- Load Model --- Specify compute units here
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU // <<< REVERT to CPU and GPU
            loadLog += "Attempting to load MLModel from final URL: \(modelToLoadURL.path)\n"
            loadLog += "Using compute units: \(config.computeUnits.description)\n"

            // Load the MLModel instance using the determined URL
            self.model = try MLModel(contentsOf: modelToLoadURL, configuration: config)

            loadLog += "Successfully loaded model: \(modelBaseName)\n"
            loadLog += "Model Description: \(self.model.modelDescription)\n"

            // --- Load Tokenizer --- Manually loading tokenizer.json and tokenizer_config.json data
            let tokenizerDataFileName = "tokenizer"
            let tokenizerConfigFileName = "tokenizer_config"
            loadLog += "Loading tokenizer by finding and reading '\(tokenizerDataFileName).json' and '\(tokenizerConfigFileName).json'...\n"

            guard let tokenizerFileURL = Bundle.main.url(forResource: tokenizerDataFileName, withExtension: "json") else {
                loadLog += "  -> FAILED: Could not find '\(tokenizerDataFileName).json' in the app bundle.\n"
                print(loadLog); throw InferenceError.tokenizerLoadFailed("Could not find '\(tokenizerDataFileName).json'.")
            }
            guard let tokenizerConfigFileURL = Bundle.main.url(forResource: tokenizerConfigFileName, withExtension: "json") else {
                loadLog += "  -> FAILED: Could not find '\(tokenizerConfigFileName).json' in the app bundle.\n"
                print(loadLog); throw InferenceError.tokenizerLoadFailed("Could not find '\(tokenizerConfigFileName).json'.")
            }
            loadLog += "  -> Found tokenizer file: \(tokenizerFileURL.path)\n"
            loadLog += "  -> Found tokenizer config file: \(tokenizerConfigFileURL.path)\n"

            do {
                let tokenizerData = try Data(contentsOf: tokenizerFileURL)
                let tokenizerConfigData = try Data(contentsOf: tokenizerConfigFileURL)
                loadLog += "  -> Successfully read data from both .json files.\n"

                // Decode the JSON data into Dictionaries first
                guard let tokenizerConfigDict = try JSONSerialization.jsonObject(with: tokenizerConfigData) as? [String: Any] else {
                    throw InferenceError.tokenizerLoadFailed("Failed to deserialize tokenizer_config.json data into a dictionary.")
                }
                guard let tokenizerJsonDict = try JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any] else {
                    throw InferenceError.tokenizerLoadFailed("Failed to deserialize tokenizer.json data into a dictionary.")
                }
                loadLog += "  -> Successfully deserialized both JSON data into dictionaries.\n"

                // Initialize Config objects from the dictionaries, casting to [NSString: Any]
                let tokenizerConfig = Config(tokenizerConfigDict as [NSString: Any])
                let tokenizerJson = Config(tokenizerJsonDict as [NSString: Any])
                loadLog += "  -> Successfully created Config objects from dictionaries.\n"

                // Attempt to initialize Tokenizer with both config and data
                // Use AutoTokenizer factory method with Config objects
                self.tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerJson)
                loadLog += "Successfully initialized Tokenizer using AutoTokenizer.from().\n"

            } catch {
                loadLog += "  -> FAILED to read or initialize tokenizer from data: \(error.localizedDescription)\n"
                print(loadLog)
                throw InferenceError.tokenizerLoadFailed("Failed to load/initialize tokenizer from data: \(error.localizedDescription)")
            }

            // Get EOS token ID directly from the tokenizer property
            self.eosTokenId = self.tokenizer.eosTokenId ?? defaultEOS
            if self.eosTokenId == defaultEOS && self.tokenizer.eosTokenId == nil {
                 loadLog += "Warning: Tokenizer did not provide an eosTokenId. Using default: \(defaultEOS)\n"
            }
            loadLog += "Using EOS token ID: \(eosTokenId)\n"

            // Print final success log
            print("--- InferenceController Initialization Log ---")
            print(loadLog)
            print("--- End Initialization Log ---")


        } catch let error as InferenceError {
             print("--- InferenceController Initialization FAILED ---")
             print(loadLog) // Print log context leading to error
             print("Error Type: InferenceError")
             print("Error Details: \(error)")
             print("--- End Initialization Failure ---")
             throw error // Re-throw InferenceError
         } catch {
             print("--- InferenceController Initialization FAILED ---")
             print(loadLog) // Print log context leading to error
             print("Error Type: Unexpected")
             print("Error Details: \(error)")
              print("--- End Initialization Failure ---")
             // Wrap other errors as InferenceError for consistency
             throw InferenceError.modelLoadingFailed("Unexpected initialization error: \(error.localizedDescription)")
         }
    }

    /// Generates a response stream based on the prompt and parameters.
    func generateResponse(prompt: String, maxTokens: Int) async throws -> AsyncStream<String> {
        // *** ADDED Hardcoded Values ***
        let temperature: Float = 0.8
        let topP: Float = 0.9

        print("Starting generation with prompt: '\(prompt)'")
        print("Parameters: Temp=\(temperature) (hardcoded), TopP=\(topP) (hardcoded), MaxTokens=\(maxTokens)")

        // --- Prepare Initial Input --- Tokenize the prompt
        // Assuming BOS token is handled by tokenizer.encode if needed for Llama 3
        // Note: encode returns [Int], not [Int]?, so direct assignment is fine.
        let promptTokens: [Int] = tokenizer.encode(text: prompt)
        guard !promptTokens.isEmpty else { // Check if encoding resulted in empty array
            throw InferenceError.inputPreparationFailed("Failed to encode prompt or resulted in empty token array.")
        }
        print("Prompt tokenized into \(promptTokens.count) IDs: \(promptTokens)")

        let promptTokenIds = promptTokens.map { Int32($0) }
        let batchSize = 1
        let initialSeqLen = promptTokenIds.count

        // Create initial input tensor (Int32)
        let initialInputIds_npy = try MLMultiArray(shape: [batchSize as NSNumber, initialSeqLen as NSNumber], dataType: .int32)
        for i in 0..<initialSeqLen {
            initialInputIds_npy[[0, i] as [NSNumber]] = promptTokenIds[i] as NSNumber
        }

        // Create initial causal mask (Float16)
        let initialCausalMask_npy = try MLMultiArray(shape: [batchSize as NSNumber, 1, initialSeqLen as NSNumber, initialSeqLen as NSNumber], dataType: .float16)
        for q in 0..<initialSeqLen {
            for k in 0..<initialSeqLen {
                initialCausalMask_npy[[0, 0, q, k] as [NSNumber]] = (k > q) ? -10000.0 : 0.0 // Large negative for masked positions
            }
        }

        // Create initial MLFeatureProvider input
        let initialInputs = try MLDictionaryFeatureProvider(dictionary: [
            "inputIds": initialInputIds_npy,
            "causalMask": initialCausalMask_npy
            // Add other inputs expected by *your specific model* if different
            // e.g., position_ids, attention_mask variants etc.
            // Check model.modelDescription.inputDescriptionsByName
        ])
        print("Initial input features prepared.")
        print("Input Descriptions:")
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            print("  - \(name): \(desc.type) \(desc.multiArrayConstraint?.shape ?? []) Optional=\(desc.isOptional)")
        }


        // --- Initialize State --- Create the initial state object
        // Use the model's makeState() method (does not throw)
        let initialPredictionState = model.makeState()
        print("Initial MLState created.")
        print("Output Descriptions:")
        for (name, desc) in model.modelDescription.outputDescriptionsByName {
             print("  - \(name): \(desc.type) \(desc.multiArrayConstraint?.shape ?? [])")
        }
        // Also print state descriptions if available
        if #available(iOS 15.0, macOS 12.0, *) { // Check availability for stateDescriptionsByName
             print("State Descriptions:")
             for (name, desc) in model.modelDescription.stateDescriptionsByName {
                 print("  - \(name): \(desc.type) \(desc.multiArrayConstraint?.shape ?? [])")
             }
         }


        // --- Run Initial Prediction --- Process the prompt first
        print("Running initial prediction for the prompt...")
        // Pass the initial state using the `using:` parameter
        let initialOutputs = try await model.prediction(from: initialInputs, using: initialPredictionState, options: MLPredictionOptions())
        print("Initial prediction completed.")
        // Note: `initialPredictionState` is now updated in-place by the call above.

        // --- Extract Initial Logits ---
        guard let initialLogitsOutput = initialOutputs.featureValue(for: "logits")?.multiArrayValue else {
            throw InferenceError.outputProcessingFailed("Could not get 'logits' from initial prediction output.")
        }
        print("Initial logits shape: \(initialLogitsOutput.shape)")
        guard initialLogitsOutput.shape.count == 3, let vocabSize = initialLogitsOutput.shape[2] as? Int else {
            throw InferenceError.outputProcessingFailed("Unexpected initial logits shape: \(initialLogitsOutput.shape)")
        }
        let lastTokenLogits = try extractLogits(from: initialLogitsOutput, sequenceIndex: initialSeqLen - 1, vocabSize: vocabSize)

        // --- Sampling for the First Token ---
        let firstPredictedTokenId: Int32
        do {
            // *** Use hardcoded values in sample call ***
            firstPredictedTokenId = try sample(logits: lastTokenLogits, temperature: temperature, topP: topP)
        } catch {
            throw InferenceError.samplingFailed("Sampling failed for the first token: \(error)")
        }
        print("Sampled first token ID: \(firstPredictedTokenId)")

        var currentTokenId = firstPredictedTokenId
        var generatedTokenIds: [Int32] = [currentTokenId]
        var currentTotalSeqLen = initialSeqLen + 1
        let currentState = initialPredictionState // Use the state object that was updated by the initial prediction

        // --- Create AsyncStream ---
        let stream = AsyncStream<String> { continuation in
            Task(priority: .userInitiated) {
                defer {
                    print("Generation loop finished.")
                    continuation.finish()
                }

                // Decode and yield the first predicted token immediately
                let firstDecodedToken = tokenizer.decode(tokens: [Int(firstPredictedTokenId)])
                if !firstDecodedToken.isEmpty {
                    continuation.yield(firstDecodedToken)
                } else {
                    print("Warning: Failed to decode or got empty string for first token ID \(firstPredictedTokenId)")
                }

                // --- Generation Loop --- Continue until maxTokens or EOS
                for i in 0..<(maxTokens - 1) { // Already generated one token
                    if currentTokenId == eosTokenId {
                        print("EOS token reached. Stopping generation.")
                        break
                    }

                    do {
                        // --- Prepare Input for Current Step ---
                        let currentInputId_npy = try MLMultiArray(shape: [1, 1], dataType: .int32)
                        currentInputId_npy[[0, 0] as [NSNumber]] = currentTokenId as NSNumber

                        let stepCausalMask_npy = try MLMultiArray(shape: [1, 1, 1, currentTotalSeqLen as NSNumber], dataType: .float16)
                        memset(stepCausalMask_npy.dataPointer, 0, stepCausalMask_npy.count * MemoryLayout<Float16>.stride)

                        let stepInputs = try MLDictionaryFeatureProvider(dictionary: [
                            "inputIds": currentInputId_npy,
                            "causalMask": stepCausalMask_npy
                             // Add other inputs if needed, matching initial step
                        ])

                        // --- Run Prediction (Stateful) --- Pass the *same* currentState object
                        // `currentState` is updated in-place by this prediction call.
                        let stepOutputs = try await model.prediction(from: stepInputs, using: currentState, options: MLPredictionOptions())

                        guard let stepLogitsOutput = stepOutputs.featureValue(for: "logits")?.multiArrayValue else {
                            throw InferenceError.outputProcessingFailed("Could not get 'logits' from step \(i+1) output.")
                        }
                        guard stepLogitsOutput.shape.count == 3, stepLogitsOutput.shape[1] == 1 else {
                             throw InferenceError.outputProcessingFailed("Unexpected step logits shape: \(stepLogitsOutput.shape)")
                        }
                        let stepLogits = try extractLogits(from: stepLogitsOutput, sequenceIndex: 0, vocabSize: vocabSize)

                        // --- Sample Next Token ---
                        // *** Use hardcoded values in sample call ***
                        let nextTokenId = try sample(logits: stepLogits, temperature: temperature, topP: topP)

                        // --- Update State for Next Iteration ---
                        currentTokenId = nextTokenId
                        generatedTokenIds.append(nextTokenId)
                        currentTotalSeqLen += 1

                        // --- Decode and Yield ---
                        let decodedToken = tokenizer.decode(tokens: [Int(nextTokenId)])
                        if !decodedToken.isEmpty {
                            continuation.yield(decodedToken)
                        } else {
                             print("Warning: Failed to decode or got empty string for token ID \(nextTokenId)")
                        }

                    } catch {
                        print("Error during generation loop step \(i+1): \(error)")
                        // --- ADDED: More detailed error logging ---
                        if let inferError = error as? InferenceError {
                            print("  -> InferenceError: \(inferError)")
                        } else if let nsError = error as NSError? {
                             print("  -> NSError Domain: \(nsError.domain), Code: \(nsError.code), UserInfo: \(nsError.userInfo)")
                        } else {
                             print("  -> Unknown error type.")
                        }
                        // --- END ADDED ---
                        let errorMessage = "Error in generation: \(error.localizedDescription)"
                        continuation.yield("\n[ERROR: \(errorMessage)]")
                        continuation.finish()
                        return // Exit the task on error
                    }
                } // End of loop

                print("Reached max tokens (\(maxTokens)) or finished naturally.")

            } // End of Task
        } // End of AsyncStream

        return stream
    }

    // --- Helper Functions ---

    /// Extracts the logits for a specific token position from the MLMultiArray.
    private func extractLogits(from logitsOutput: MLMultiArray, sequenceIndex: Int, vocabSize: Int) throws -> [Float] {
        guard logitsOutput.dataType == .float16 || logitsOutput.dataType == .float32 else {
            throw InferenceError.outputProcessingFailed("Logits MLMultiArray has unexpected dataType: \(logitsOutput.dataType)")
        }
        let seqLen = logitsOutput.shape[1].intValue
        guard sequenceIndex < seqLen else {
            throw InferenceError.outputProcessingFailed("Sequence index \(sequenceIndex) out of bounds for logits seqLen \(seqLen)")
        }

        var logits: [Float] = Array(repeating: 0.0, count: vocabSize)
        let pointer: UnsafeMutableRawPointer = logitsOutput.dataPointer
        let seqOffset = sequenceIndex * vocabSize

        if logitsOutput.dataType == .float16 {
            let float16Ptr = pointer.bindMemory(to: Float16.self, capacity: logitsOutput.count)
            for i in 0..<vocabSize {
                logits[i] = Float(float16Ptr[seqOffset + i])
            }
        } else { // Float32
            let float32Ptr = pointer.bindMemory(to: Float.self, capacity: logitsOutput.count)
            for i in 0..<vocabSize {
                logits[i] = float32Ptr[seqOffset + i]
            }
        }
        return logits
    }

    /// Applies temperature scaling and top-p sampling to logits to select the next token.
    private func sample(logits: [Float], temperature: Float, topP: Float) throws -> Int32 {
        guard !logits.isEmpty else {
             throw InferenceError.samplingFailed("Logits array is empty.")
        }
        var probabilities = logits
        if temperature > 0.0 {
            probabilities = probabilities.map { $0 / temperature }
        } else {
            if let maxIndex = logits.indices.max(by: { logits[$0] < logits[$1] }) {
                return Int32(maxIndex)
            } else {
                throw InferenceError.samplingFailed("Could not find argmax for temperature=0.")
            }
        }

        // Apply Softmax (stable version)
        let maxLogit = probabilities.max() ?? 0
        let expProbs = probabilities.map { exp($0 - maxLogit) }
        let sumExpProbs = expProbs.reduce(0, +)
        guard sumExpProbs > Float.ulpOfOne else {
             print("Warning: Sum of probabilities after softmax near zero (\(sumExpProbs)). Falling back to argmax.")
             if let maxIndex = logits.indices.max(by: { logits[$0] < logits[$1] }) {
                 return Int32(maxIndex)
             } else {
                 throw InferenceError.samplingFailed("Could not find argmax fallback after softmax failure.")
             }
        }
        probabilities = expProbs.map { $0 / sumExpProbs }

        // Apply Top-P (Nucleus Sampling)
        if topP > 0.0 && topP < 1.0 {
            let sortedIndices = probabilities.indices.sorted { probabilities[$0] > probabilities[$1] }
            let sortedProbs = sortedIndices.map { probabilities[$0] }
            var cumulativeProbs: [Float] = []
            var currentSum: Float = 0
            for p in sortedProbs { currentSum += p; cumulativeProbs.append(currentSum) }
            let cutoffIndex = cumulativeProbs.firstIndex { $0 >= topP } ?? (sortedProbs.count - 1)
            var mask = [Bool](repeating: true, count: probabilities.count)
            for i in 0...cutoffIndex { mask[sortedIndices[i]] = false }
            for i in mask.indices where mask[i] { probabilities[i] = 0.0 }
            let remainingSum = probabilities.reduce(0, +)
            if remainingSum > Float.ulpOfOne {
                probabilities = probabilities.map { $0 / remainingSum }
            } else {
                 print("Warning: Sum of probabilities after Top-P near zero (\(remainingSum)). Falling back to highest prob token in nucleus.")
                 return Int32(sortedIndices[0])
            }
        }

        // Weighted Random Choice
        let randomValue = Float.random(in: 0..<1)
        var cumulativeSum: Float = 0
        for (index, prob) in probabilities.enumerated() {
            cumulativeSum += prob
            if randomValue < cumulativeSum {
                return Int32(index)
            }
        }
        print("Warning: Sampling fallback - choosing last index.")
        return Int32(probabilities.count - 1)
    }
}

// Helper extension for MLModelConfiguration computeUnits description
extension MLComputeUnits {
    var description: String {
        switch self {
        case .all: return "all"
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        @unknown default:
            return "unknown"
        }
    }
}

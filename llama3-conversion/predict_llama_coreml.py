#hi
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
import time
import logging
import sys
import os

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.2-1B"  # Make sure this matches the converted model
# Use the output path defined in the conversion script
# DEFAULT_MODEL_PATH = "Llama3Stateful_Pal4_LUT8.mlpackage" # Example for compressed
DEFAULT_MODEL_PATH = "Llama1Stateful_Pal4_LUT8.mlpackage" # Example for FP16
MAX_NEW_TOKENS = 100 # Limit the number of tokens to generate
EOS_TOKEN_ID = 128001 # Llama 3 EOS token ID (often 128001 or 128009, verify with tokenizer)
TEMPERATURE = 0.8 # Add temperature for sampling (0.0 = argmax, > 0 introduces randomness)
TOP_P = 0.9       # Add nucleus sampling (e.g., 0.9 means consider tokens comprising 90% of probability mass)

# --- Logging Setup ---
logger = logging.getLogger("llama3_predict")
logger.setLevel(logging.INFO)
fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
formatter = logging.Formatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


# Causal mask creation is handled within run_prediction based on context

def run_stateful_generation(model_path: str = DEFAULT_MODEL_PATH, prompt: str = "what is artificial intelligence?", max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE, top_p: float = TOP_P):
    """ Loads the Core ML model, runs stateful generation for a prompt, and decodes the sequence. """
    logger.info(f"Attempting to load Core ML model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model package not found at {model_path}. Did the conversion script run successfully?")
        return

    try:
        # Load the Core ML model package
        # Specify compute units if needed (e.g., CPU_AND_GPU or ALL)
        # Using ALL is recommended for stateful models based on testing
        start_load_time = time.time()
        mlmodel = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
        load_time = time.time() - start_load_time
        logger.info(f"Successfully loaded Core ML model in {load_time:.2f} seconds.")

        # Load the corresponding Hugging Face tokenizer
        logger.info(f"Loading tokenizer for {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # Determine EOS token ID from tokenizer if not hardcoded
        eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id else EOS_TOKEN_ID
        logger.info(f"Using EOS token ID: {eos_token_id}")
        logger.info("Tokenizer loaded.")

        # --- Prepare Input Data for Initial Prompt ---
        logger.info(f"Tokenizing prompt: \"{prompt}\" ...")
        # Note: Ensure tokenizer adds BOS token if the model expects it. Llama often does.
        # Add add_special_tokens=True if BOS/EOS are needed for the specific checkpoint.
        input_ids = tokenizer.encode(prompt, return_tensors="np", add_special_tokens=True)
        batch_size, seq_len = input_ids.shape
        logger.info(f"Input IDs shape: {input_ids.shape}")

        # Create the upper triangular matrix of ones (k=1 shifts the diagonal)
        upper_triangular = np.triu(np.ones((seq_len, seq_len), dtype=np.float16), k=1)
        # Multiply by a large negative number (or -np.inf if supported and stable)
        initial_causal_mask = upper_triangular * -10000.0 # Use a large negative float
        # Add batch and head dimensions (1, 1, seq_len, seq_len)
        initial_causal_mask = np.expand_dims(initial_causal_mask, axis=(0, 1))
        logger.info(f"Initial causal mask shape: {initial_causal_mask.shape}")

        # Prepare the input dictionary for Core ML prediction
        coreml_inputs = {
            "inputIds": input_ids.astype(np.int32), # Ensure int32 type
            "causalMask": initial_causal_mask
        }

        # --- Initialize State ---
        logger.info("Creating initial model state...")
        state = mlmodel.make_state()
        logger.info("Initial state created.")

        # --- Run Prediction for Initial Prompt ---
        logger.info("Running prediction for the initial prompt...")
        start_pred_time = time.time()
        outputs = mlmodel.predict(coreml_inputs, state=state)
        pred_time = time.time() - start_pred_time
        logger.info(f"Initial prediction complete in {pred_time:.3f} seconds.")

        # --- Process Initial Output ---
        if "logits" not in outputs:
            logger.error("Prediction failed: 'logits' not found in output dictionary.")
            return

        logits = outputs["logits"] # Shape: (batch_size, seq_len, vocab_size)
        logger.info(f"Output logits shape: {logits.shape}")

        # Get logits for the *last* token in the input sequence
        next_token_logits = logits[0, -1, :]

        # Apply temperature to the first prediction's logits as well
        if temperature > 0.0:
            scaled_logits = next_token_logits / temperature
            # Apply softmax
            probs = np.exp(scaled_logits - np.max(scaled_logits)) # Softmax (stable)
            probs /= np.sum(probs)

            # Apply Top-P (nucleus sampling)
            if top_p < 1.0 and top_p > 0.0: # Only apply if top_p is in (0, 1)
                sorted_indices = np.argsort(probs)[::-1]
                cumulative_probs = np.cumsum(probs[sorted_indices])

                # Find indices of tokens to keep (nucleus)
                indices_to_remove = cumulative_probs > top_p
                # Shift the mask: keep the first token whose cumulative prob exceeds top_p
                indices_to_remove[1:] = indices_to_remove[:-1].copy()
                indices_to_remove[0] = False # Always keep the highest prob token

                indices_removed = sorted_indices[indices_to_remove]
                probs[indices_removed] = 0.0 # Zero out probabilities outside the nucleus

                # Renormalize, handling potential zero sum
                sum_probs = np.sum(probs)
                if sum_probs > 1e-9: # Check if sum is significantly larger than zero
                    probs /= sum_probs
                else:
                    # Fallback: If nucleus is empty or near-zero probability (should be rare),
                    # just revert to selecting the single token with the highest original probability.
                    logger.warning(f"Sum of probabilities after Top-P is {sum_probs:.4e}. Falling back to argmax for this step.")
                    probs = np.zeros_like(probs)
                    # Use the logits *before* temperature scaling for the true argmax
                    probs[np.argmax(next_token_logits)] = 1.0

            # Sample from the (potentially modified) distribution
            predicted_token_id = np.random.choice(len(probs), p=probs)
            logger.info(f"Sampled first token ID (temp={temperature}, top_p={top_p}): {predicted_token_id}")
        else: # Use argmax if temperature is 0
             logger.info(f"Predicted first token ID (argmax): {predicted_token_id}")

        # --- Stateful Generation Loop ---
        generated_token_ids = [predicted_token_id] # Start with the first predicted token
        current_input_ids = np.array([[predicted_token_id]], dtype=np.int32)
        current_total_seq_len = seq_len + 1 # Prompt length + 1 generated token

        logger.info("--- Starting stateful generation loop ---")
        loop_start_time = time.time()

        for i in range(max_new_tokens - 1): # Already generated one token
            # Prepare causal mask for single token generation step
            # Shape: (1, 1, 1, current_total_seq_len)
            # As per Llama 3.1 doc: all zeros for decoding phase
            step_causal_mask = np.zeros((1, 1, 1, current_total_seq_len), dtype=np.float16)

            step_inputs = {
                "inputIds": current_input_ids,
                "causalMask": step_causal_mask
            }

            # Run prediction, updating state in-place
            outputs = mlmodel.predict(step_inputs, state=state)

            # Process output
            logits = outputs["logits"] # Shape: (1, 1, vocab_size)
            next_token_logits = logits[0, -1, :] # Logits for the single new token

            # Apply temperature scaling and sampling
            if temperature > 0.0:
                scaled_logits = next_token_logits / temperature
                # Apply softmax
                probs = np.exp(scaled_logits - np.max(scaled_logits)) # Softmax (stable)
                probs /= np.sum(probs)

                # Apply Top-P (nucleus sampling)
                if top_p < 1.0 and top_p > 0.0:
                    sorted_indices = np.argsort(probs)[::-1]
                    cumulative_probs = np.cumsum(probs[sorted_indices])

                    # Find indices of tokens to keep (nucleus)
                    indices_to_remove = cumulative_probs > top_p
                    # Shift the mask: keep the first token whose cumulative prob exceeds top_p
                    indices_to_remove[1:] = indices_to_remove[:-1].copy()
                    indices_to_remove[0] = False # Always keep the highest prob token

                    indices_removed = sorted_indices[indices_to_remove]
                    probs[indices_removed] = 0.0 # Zero out probabilities outside the nucleus

                    # Renormalize
                    sum_probs = np.sum(probs)
                    if sum_probs > 1e-9:
                        probs /= sum_probs
                    else:
                        logger.warning(f"Sum of probabilities after Top-P is {sum_probs:.4e}. Falling back to argmax for this step.")
                        probs = np.zeros_like(probs)
                        probs[np.argmax(next_token_logits)] = 1.0

                # Sample from the distribution
                predicted_token_id = np.random.choice(len(probs), p=probs)
            else:
                # Fallback to argmax if temperature is 0
                predicted_token_id = np.argmax(next_token_logits)

            # Check for EOS token
            if predicted_token_id == eos_token_id:
                logger.info(f"EOS token ({eos_token_id}) generated. Stopping generation.")
                break

            generated_token_ids.append(predicted_token_id)
            current_input_ids = np.array([[predicted_token_id]], dtype=np.int32)
            current_total_seq_len += 1

            # Optional: Log progress periodically
            if (i + 1) % 10 == 0:
                 logger.info(f"Generated {i + 2}/{max_new_tokens} tokens...")

        loop_time = time.time() - loop_start_time
        num_generated = len(generated_token_ids)
        logger.info(f"--- Generation loop finished ---")
        logger.info(f"Generated {num_generated} tokens in {loop_time:.3f} seconds ({num_generated / loop_time:.2f} tokens/sec)")

        # --- Decode the full generated sequence ---
        # Combine prompt IDs and generated IDs
        full_sequence_ids = np.concatenate([input_ids[0], np.array(generated_token_ids)], axis=0)
        # Decode the whole sequence (or just the generated part)
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(full_sequence_ids, skip_special_tokens=True) # Includes prompt

        print("\n--- Generation Result ---")
        print(f"Prompt: {prompt}")
        print(f"\nGenerated Text:\n{generated_text}")
        # print(f"\nFull Text (incl. prompt):\n{full_text}") # Optional: print everything
        print("\n-------------------------")

    except Exception as e:
        logger.error("An error occurred during prediction:", exc_info=True)

if __name__ == "__main__":
    # You can change the model path or prompt via command-line arguments or by editing here
    model_to_test = DEFAULT_MODEL_PATH
    test_prompt = "what is artificial intelligence?" # New test prompt
    run_stateful_generation(model_path=model_to_test, prompt=test_prompt, temperature=TEMPERATURE, top_p=TOP_P) 



# #hi
# import coremltools as ct
# import numpy as np
# from transformers import AutoTokenizer
# import time
# import logging
# import sys
# import os

# # --- Configuration ---
# MODEL_ID = "meta-llama/Llama-3.2-3B"  # Make sure this matches the converted model
# # Use the output path defined in the conversion script
# # DEFAULT_MODEL_PATH = "Llama3Stateful_Pal4_LUT8.mlpackage" # Example for compressed
# DEFAULT_MODEL_PATH = "Llama3Stateful_Pal4_LUT8.mlpackage" # Example for FP16
# MAX_NEW_TOKENS = 100 # Limit the number of tokens to generate
# EOS_TOKEN_ID = 128001 # Llama 3 EOS token ID (often 128001 or 128009, verify with tokenizer)
# TEMPERATURE = 0.8 # Add temperature for sampling (0.0 = argmax, > 0 introduces randomness)

# # --- Logging Setup ---
# logger = logging.getLogger("llama3_predict")
# logger.setLevel(logging.INFO)
# fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
# formatter = logging.Formatter(fmt)
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.INFO)
# ch.setFormatter(formatter)
# logger.addHandler(ch)


# # Causal mask creation is handled within run_prediction based on context

# def run_stateful_generation(model_path: str = DEFAULT_MODEL_PATH, prompt: str = "what is artificial intelligence?", max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE):
#     """ Loads the Core ML model, runs stateful generation for a prompt, and decodes the sequence. """
#     logger.info(f"Attempting to load Core ML model from: {model_path}")
#     if not os.path.exists(model_path):
#         logger.error(f"Model package not found at {model_path}. Did the conversion script run successfully?")
#         return

#     try:
#         # Load the Core ML model package
#         # Specify compute units if needed (e.g., CPU_AND_GPU or ALL)
#         # Using ALL is recommended for stateful models based on testing
#         start_load_time = time.time()
#         mlmodel = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
#         load_time = time.time() - start_load_time
#         logger.info(f"Successfully loaded Core ML model in {load_time:.2f} seconds.")

#         # Load the corresponding Hugging Face tokenizer
#         logger.info(f"Loading tokenizer for {MODEL_ID}...")
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#         # Determine EOS token ID from tokenizer if not hardcoded
#         eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id else EOS_TOKEN_ID
#         logger.info(f"Using EOS token ID: {eos_token_id}")
#         logger.info("Tokenizer loaded.")

#         # --- Prepare Input Data for Initial Prompt ---
#         logger.info(f"Tokenizing prompt: \"{prompt}\" ...")
#         # Note: Ensure tokenizer adds BOS token if the model expects it. Llama often does.
#         # Add add_special_tokens=True if BOS/EOS are needed for the specific checkpoint.
#         input_ids = tokenizer.encode(prompt, return_tensors="np", add_special_tokens=True)
#         batch_size, seq_len = input_ids.shape
#         logger.info(f"Input IDs shape: {input_ids.shape}")

#         # Create the upper triangular matrix of ones (k=1 shifts the diagonal)
#         upper_triangular = np.triu(np.ones((seq_len, seq_len), dtype=np.float16), k=1)
#         # Multiply by a large negative number (or -np.inf if supported and stable)
#         initial_causal_mask = upper_triangular * -10000.0 # Use a large negative float
#         # Add batch and head dimensions (1, 1, seq_len, seq_len)
#         initial_causal_mask = np.expand_dims(initial_causal_mask, axis=(0, 1))
#         logger.info(f"Initial causal mask shape: {initial_causal_mask.shape}")

#         # Prepare the input dictionary for Core ML prediction
#         coreml_inputs = {
#             "inputIds": input_ids.astype(np.int32), # Ensure int32 type
#             "causalMask": initial_causal_mask
#         }

#         # --- Initialize State ---
#         logger.info("Creating initial model state...")
#         state = mlmodel.make_state()
#         logger.info("Initial state created.")

#         # --- Run Prediction for Initial Prompt ---
#         logger.info("Running prediction for the initial prompt...")
#         start_pred_time = time.time()
#         outputs = mlmodel.predict(coreml_inputs, state=state)
#         pred_time = time.time() - start_pred_time
#         logger.info(f"Initial prediction complete in {pred_time:.3f} seconds.")

#         # --- Process Initial Output ---
#         if "logits" not in outputs:
#             logger.error("Prediction failed: 'logits' not found in output dictionary.")
#             return

#         logits = outputs["logits"] # Shape: (batch_size, seq_len, vocab_size)
#         logger.info(f"Output logits shape: {logits.shape}")

#         # Get logits for the *last* token in the input sequence
#         next_token_logits = logits[0, -1, :]

#         # Find the token ID with the highest probability (argmax)
#         predicted_token_id = np.argmax(next_token_logits)
#         logger.info(f"Predicted first token ID: {predicted_token_id}")

#         # Apply temperature scaling and sampling
#         if temperature > 0.0:
#             scaled_logits = next_token_logits / temperature
#             # Apply softmax to get probabilities
#             probs = np.exp(scaled_logits - np.max(scaled_logits)) # Softmax (stable)
#             probs /= np.sum(probs)
#             # Sample from the distribution
#             predicted_token_id = np.random.choice(len(probs), p=probs)
#             logger.info(f"Sampled first token ID (temp={temperature}): {predicted_token_id}")
#         else: # Use argmax if temperature is 0
#              logger.info(f"Predicted first token ID (argmax): {predicted_token_id}")

#         # --- Stateful Generation Loop ---
#         generated_token_ids = [predicted_token_id] # Start with the first predicted token
#         current_input_ids = np.array([[predicted_token_id]], dtype=np.int32)
#         current_total_seq_len = seq_len + 1 # Prompt length + 1 generated token

#         logger.info("--- Starting stateful generation loop ---")
#         loop_start_time = time.time()

#         for i in range(max_new_tokens - 1): # Already generated one token
#             # Prepare causal mask for single token generation step
#             # Shape: (1, 1, 1, current_total_seq_len)
#             # As per Llama 3.1 doc: all zeros for decoding phase
#             step_causal_mask = np.zeros((1, 1, 1, current_total_seq_len), dtype=np.float16)

#             step_inputs = {
#                 "inputIds": current_input_ids,
#                 "causalMask": step_causal_mask
#             }

#             # Run prediction, updating state in-place
#             outputs = mlmodel.predict(step_inputs, state=state)

#             # Process output
#             logits = outputs["logits"] # Shape: (1, 1, vocab_size)
#             next_token_logits = logits[0, -1, :] # Logits for the single new token

#             # Apply temperature scaling and sampling
#             if temperature > 0.0:
#                 scaled_logits = next_token_logits / temperature
#                 # Apply softmax to get probabilities
#                 probs = np.exp(scaled_logits - np.max(scaled_logits)) # Softmax (stable)
#                 probs /= np.sum(probs)
#                 # Sample from the distribution
#                 predicted_token_id = np.random.choice(len(probs), p=probs)
#             else:
#                 # Fallback to argmax if temperature is 0
#                 predicted_token_id = np.argmax(next_token_logits)

#             # Check for EOS token
#             if predicted_token_id == eos_token_id:
#                 logger.info(f"EOS token ({eos_token_id}) generated. Stopping generation.")
#                 break

#             generated_token_ids.append(predicted_token_id)
#             current_input_ids = np.array([[predicted_token_id]], dtype=np.int32)
#             current_total_seq_len += 1

#             # Optional: Log progress periodically
#             if (i + 1) % 10 == 0:
#                  logger.info(f"Generated {i + 2}/{max_new_tokens} tokens...")

#         loop_time = time.time() - loop_start_time
#         num_generated = len(generated_token_ids)
#         logger.info(f"--- Generation loop finished ---")
#         logger.info(f"Generated {num_generated} tokens in {loop_time:.3f} seconds ({num_generated / loop_time:.2f} tokens/sec)")

#         # --- Decode the full generated sequence ---
#         # Combine prompt IDs and generated IDs
#         full_sequence_ids = np.concatenate([input_ids[0], np.array(generated_token_ids)], axis=0)
#         # Decode the whole sequence (or just the generated part)
#         generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
#         full_text = tokenizer.decode(full_sequence_ids, skip_special_tokens=True) # Includes prompt

#         print("\n--- Generation Result ---")
#         print(f"Prompt: {prompt}")
#         print(f"\nGenerated Text:\n{generated_text}")
#         # print(f"\nFull Text (incl. prompt):\n{full_text}") # Optional: print everything
#         print("\n-------------------------")

#     except Exception as e:
#         logger.error("An error occurred during prediction:", exc_info=True)

# if __name__ == "__main__":
#     # You can change the model path or prompt via command-line arguments or by editing here
#     model_to_test = DEFAULT_MODEL_PATH
#     test_prompt = "what is artificial intelligence?" # New test prompt
#     run_stateful_generation(model_path=model_to_test, prompt=test_prompt, temperature=TEMPERATURE) 


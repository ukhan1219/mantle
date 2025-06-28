import time
start_time = time.perf_counter()

import_start = time.perf_counter()
import coremltools as ct
import numpy as np
import torch # Needed for dtypes and tokenizer interaction
from transformers import AutoTokenizer
import argparse
import sys
import logging
import_end = time.perf_counter()

# ──── Logging Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("mistral_prediction")
logger.setLevel(logging.INFO) # Start with INFO level

fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
formatter = logging.Formatter(fmt)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(f"Imports completed in {import_end - import_start:.2f} seconds")

# ──── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "../prospective-models/MistralStateful_Pal4_LUT8.mlpackage"
TOKENIZER_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_PROMPT = "Recommend a place to visit in Seattle in June."
DEFAULT_MAX_TOKENS = 100


# ──── Generation Function ───────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_new_tokens: int):
    """
    Generates text autoregressively using the stateful Core ML model.
    """
    generation_start = time.perf_counter()
    
    logger.info(f"Tokenizing prompt...")
    tokenize_start = time.perf_counter()
    # Note: Mistral instruct models expect specific formatting, including BOS token.
    # We'll add bos but skip generation_prompt formatting for this basic example.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors=None)
    tokenize_end = time.perf_counter()
    logger.info(f"Tokenization completed in {tokenize_end - tokenize_start:.2f} seconds")
    
    if not prompt_ids:
        logger.error("Tokenizer returned empty list for the prompt.")
        return "Error: Invalid prompt or tokenizer issue."
    logger.debug(f"Prompt token IDs: {prompt_ids}")

    generated_ids = list(prompt_ids) # Start with the prompt IDs
    prompt_len = len(prompt_ids)

    # --- Initial Prediction Step (Process the whole prompt) ---
    logger.info("Running initial prediction for the prompt...")
    initial_pred_start = time.perf_counter()
    initial_input_ids = np.array([prompt_ids], dtype=np.int32) # Shape (1, P)
    # Mask shape (B=1, 1, Q=P, K=P) - Use ones, assuming state handles causality
    initial_causal_mask = np.ones((1, 1, prompt_len, prompt_len), dtype=np.float16)

    try:
        logger.debug(f"Initial predict input shapes: inputIds={initial_input_ids.shape}, causalMask={initial_causal_mask.shape}")
        initial_predictions = model.predict({
            'inputIds': initial_input_ids,
            'causalMask': initial_causal_mask
        })
        initial_logits = initial_predictions['logits'] # Shape (1, P, V)
        initial_pred_end = time.perf_counter()
        logger.info(f"Initial prediction completed in {initial_pred_end - initial_pred_start:.2f} seconds")
        logger.debug(f"Initial predict output logits shape: {initial_logits.shape}")

        # Get the logits for the *last* token of the prompt to predict the first new token
        next_token_logits = initial_logits[:, -1, :] # Shape (1, V)
        next_token_id = np.argmax(next_token_logits).item()

        logger.debug(f"First generated token ID: {next_token_id}")
        if next_token_id == tokenizer.eos_token_id:
            logger.info("EOS token predicted right after prompt.")
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

        generated_ids.append(next_token_id)

    except Exception as e:
        logger.error(f"Error during initial prediction: {e}", exc_info=True)
        return f"Error during initial prediction: {e}"

    # --- Autoregressive Loop ---
    logger.info(f"Starting generation loop (max_new_tokens={max_new_tokens})...")
    loop_start = time.perf_counter()
    tokens_generated = 0
    for i in range(max_new_tokens):
        token_start = time.perf_counter()
        current_input_id = np.array([[next_token_id]], dtype=np.int32) # Shape (1, 1)
        seq_len = len(generated_ids) # Current total sequence length

        # Mask shape (B=1, 1, Q=1, K=seq_len)
        loop_causal_mask = np.ones((1, 1, 1, seq_len), dtype=np.float16)

        try:
            logger.debug(f"Loop step {i+1}: predict input shapes: inputIds={current_input_id.shape}, causalMask={loop_causal_mask.shape}")
            predictions = model.predict({
                'inputIds': current_input_id,
                'causalMask': loop_causal_mask
            })
            logits = predictions['logits'] # Shape (1, 1, V)
            token_end = time.perf_counter()
            logger.debug(f"Loop step {i+1}: Generated token in {token_end - token_start:.4f} seconds")
            logger.debug(f"Loop step {i+1}: predict output logits shape: {logits.shape}")

            next_token_id = np.argmax(logits[:, -1, :]).item() # Get the ID for the new token
            tokens_generated += 1
            logger.debug(f"Loop step {i+1}: Generated token ID: {next_token_id}")

            if next_token_id == tokenizer.eos_token_id:
                logger.info(f"EOS token encountered after generating {tokens_generated} tokens.")
                break

            generated_ids.append(next_token_id)

        except Exception as e:
            logger.error(f"Error during generation loop step {i+1}: {e}", exc_info=True)
            return f"Error during generation loop: {e} ... (partial result: {tokenizer.decode(generated_ids, skip_special_tokens=True)})"

    loop_end = time.perf_counter()
    logger.info(f"Generation loop finished after {tokens_generated} tokens in {loop_end - loop_start:.2f} seconds")
    logger.info(f"Average time per token: {(loop_end - loop_start) / tokens_generated:.4f} seconds")

    # --- Decode ---
    logger.info("Decoding final generated sequence...")
    decode_start = time.perf_counter()
    final_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    decode_end = time.perf_counter()
    logger.info(f"Decoding completed in {decode_end - decode_start:.2f} seconds")
    logger.debug(f"Decoded text: {final_text}")

    generation_end = time.perf_counter()
    logger.info(f"Total generation time: {generation_end - generation_start:.2f} seconds")

    return final_text


# ──── Main Execution ────────────────────────────────────────────────────────────

def main():
    main_start = time.perf_counter()
    
    parser = argparse.ArgumentParser(description="Generate text using a stateful Mistral Core ML model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the .mlpackage model file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Input prompt for the model (default: '{DEFAULT_PROMPT}')",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of new tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--use-compiled",
        action="store_true",
        help="Load from a compiled .mlmodelc directory (if it exists) for potentially faster subsequent loads.",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    # --- Load Model ---
    model_load_path = args.model_path
    load_start_time = time.perf_counter()
    logger.info(f"Loading Core ML model from: {model_load_path}...")

    # Optional: Use CompiledMLModel if available and requested
    compiled_path = model_load_path.replace(".mlpackage", ".mlmodelc")
    if args.use_compiled:
        try:
            # Try loading compiled model first
            logger.info(f"Attempting to load compiled model from: {compiled_path}...")
            compile_start = time.perf_counter()
            mlmodel = ct.models.CompiledMLModel(compiled_path, compute_units=ct.ComputeUnit.ALL)
            compile_end = time.perf_counter()
            logger.info(f"Successfully loaded compiled model in {compile_end - compile_start:.2f} seconds")
        except Exception as e:
            logger.warning(f"Failed to load compiled model from {compiled_path}: {e}. Falling back to .mlpackage.")
            fallback_start = time.perf_counter()
            mlmodel = ct.models.MLModel(model_load_path, compute_units=ct.ComputeUnit.ALL)
            fallback_end = time.perf_counter()
            logger.info(f"Loaded uncompiled model in {fallback_end - fallback_start:.2f} seconds")
    else:
        regular_load_start = time.perf_counter()
        mlmodel = ct.models.MLModel(model_load_path, compute_units=ct.ComputeUnit.ALL)
        regular_load_end = time.perf_counter()
        logger.info(f"Loaded model in {regular_load_end - regular_load_start:.2f} seconds")

    load_end_time = time.perf_counter()
    logger.info(f"Total model loading time: {load_end_time - load_start_time:.2f} seconds")
    logger.debug(f"Model description:{mlmodel.get_spec().description}")

    # --- Load Tokenizer ---
    tokenizer_load_start = time.perf_counter()
    logger.info(f"Loading tokenizer: {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        pass # Keep it simple for now for single generation.
    tokenizer_load_end = time.perf_counter()
    logger.info(f"Tokenizer loaded in {tokenizer_load_end - tokenizer_load_start:.2f} seconds")

    # --- Generate Text ---
    generation_start_time = time.perf_counter()
    generated_text = generate(mlmodel, tokenizer, args.prompt, args.max_new_tokens)
    generation_end_time = time.perf_counter()

    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")
    logger.info(f"Total generation time: {generation_end_time - generation_start_time:.2f} seconds")
    
    main_end = time.perf_counter()
    logger.info(f"Total script execution time: {main_end - start_time:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Unhandled exception during prediction:", exc_info=True)
        sys.exit(1) 
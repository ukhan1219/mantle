import os
import sys
import logging
import traceback

import torch
from transformers import LlamaForCausalLM, AutoConfig
import coremltools as ct
import numpy as np

# ──── Logging Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("llama3_conversion") # Logger name reflects Llama 3
logger.setLevel(logging.DEBUG)

fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
formatter = logging.Formatter(fmt)

# Console handler for high-level info
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler for detailed debug
fh = logging.FileHandler("llama3_conversion.log", mode="w", encoding="utf-8") # Log file name reflects Llama 3
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# ──── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.2-1B"
BATCH_SIZE = 1
SEQUENCE_LENGTH = 128  # Fixed sequence length for naive conversion

# Set up device and dtype
USE_CPU = False  # Try MPS first
if not USE_CPU and torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS device for conversion")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA device for conversion")
else:
    device = torch.device("cpu")
    USE_CPU = True
    logger.info("MPS/CUDA not available or USE_CPU=True, falling back to CPU.")

dtype = torch.float16
logger.info(f"Using device: {device}, dtype: {dtype}")


# --- Simple Core ML Validation Function ---
def validate_coreml_model(mlmodel: ct.models.MLModel, batch_size: int, sequence_length: int, vocab_size: int):
    """ Perform a basic prediction using the Core ML model to validate conversion. """
    logger.info("--- Starting Core ML Model Validation ---")
    try:
        # Create dummy input data
        input_ids_np = np.zeros((batch_size, sequence_length), dtype=np.int32)
        
        # Create input dictionary
        coreml_input_dict = {"inputIds": input_ids_np}
        logger.debug(f"Validation input dict prepared: {coreml_input_dict.keys()}")

        # Run prediction
        logger.debug("Running Core ML prediction for validation...")
        output_dict = mlmodel.predict(coreml_input_dict)
        logger.debug(f"Core ML prediction successful. Output keys: {output_dict.keys()}")

        # Check output presence and shape
        if "logits" not in output_dict:
            logger.error("Validation Failed: 'logits' key not found in prediction output.")
            return False

        logits_output = output_dict["logits"]
        expected_logits_shape = (batch_size, sequence_length, vocab_size)
        actual_shape = logits_output.shape
        logger.debug(f"Validation output logits shape: {actual_shape}")

        if actual_shape != expected_logits_shape:
             logger.error(f"Validation Failed: Unexpected logits shape. Expected {expected_logits_shape}, Got {actual_shape}")
             return False
        else:
             logger.info(f"Validation output logits shape matches expected shape: {expected_logits_shape}")

        logger.info("--- Core ML Model Validation Successful ---")
        return True

    except Exception as e:
        logger.error(f"--- Core ML Model Validation Failed ---")
        logger.error(f"Error during validation prediction: {e}\n{traceback.format_exc()}")
        return False

# --- Main Conversion Script ---
def main():
    logger.info(f"Starting Llama 3.2 ({MODEL_ID}) naive conversion process...")
    
    # Load the model directly without any wrapper
    logger.info(f"Loading LlamaForCausalLM model: {MODEL_ID}...")
    config = AutoConfig.from_pretrained(MODEL_ID)
    logger.info(f"Model config loaded:")
    logger.info(f"  - num_hidden_layers: {config.num_hidden_layers}")
    logger.info(f"  - num_attention_heads: {config.num_attention_heads}")
    logger.info(f"  - hidden_size: {config.hidden_size}")
    logger.info(f"  - vocab_size: {config.vocab_size}")

    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # Use eager attention for tracing
    ).to(device).eval()
    logger.info(f"Model {MODEL_ID} loaded successfully onto {device}.")

    # Create example input for tracing
    example_input_ids = torch.zeros((BATCH_SIZE, SEQUENCE_LENGTH), dtype=torch.int32, device=device)
    logger.info(f"Example input_ids shape: {example_input_ids.shape}, dtype: {example_input_ids.dtype}")

    # Trace the model
    logger.info("Tracing the model with torch.jit.trace...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model,
                example_input_ids,
                check_trace=False
            )
        logger.info("Tracing complete.")
    except Exception as e:
        logger.error(f"Tracing failed: {e}", exc_info=True)
        sys.exit(1)

    # Define Core ML input and output types
    inputs = [
        ct.TensorType(
            name="inputIds",
            shape=(BATCH_SIZE, ct.RangeDim(lower_bound=1, upper_bound=512, default=SEQUENCE_LENGTH)),
            dtype=np.int32
        )
    ]
    outputs = [ct.TensorType(name="logits", dtype=np.float16)]
    
    logger.info("Core ML input and output types defined.")

    # Convert to Core ML
    logger.info("Converting traced model to Core ML...")
    try:
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=ct.target.iOS16,  # Use iOS16 for naive conversion
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL
        )
        logger.info("Core ML conversion successful.")
    except Exception as e:
        logger.error(f"Core ML conversion failed: {e}", exc_info=True)
        sys.exit(1)

    # Validate the model
    if not validate_coreml_model(mlmodel, BATCH_SIZE, SEQUENCE_LENGTH, config.vocab_size):
        logger.error("Model validation failed. Exiting.")
        sys.exit(1)

    # --- Save the validated FP16 model (Intermediate Step) ---
    fp16_filename = "Llama1Stateful_FP16.mlpackage"
    fp16_output_path = os.path.join(os.path.dirname(__file__) or ".", fp16_filename)
    logger.info(f"Saving validated FP16 Core ML model to {fp16_output_path}...")
    try:
        mlmodel.save(fp16_output_path)
        logger.info(f"Successfully saved FP16 model to {fp16_output_path}")
    except Exception as e:
        logger.error(f"Failed to save FP16 model: {e}", exc_info=True)
        # Continue to compression anyway, but log the failure
        logger.warning("Continuing with compression despite FP16 save failure.")


    # --- Add Metadata ---
    logger.info("Adding metadata to the Core ML model...")
    mlmodel.author  = "Meta & Core ML Tools User" # Updated author
    mlmodel.license = "Llama 3.2 Community License" # Check correct license name
    mlmodel.version = "1.0"
    mlmodel.short_description = (
        f"Stateful Core ML conversion of {MODEL_ID} with FP16 precision initially, "
        f"{CONTEXT_LENGTH}-token context, intended for 4-bit Palettization + 8-bit LUT quantization."
    )
    # Set Hugging Face model ID for potential use in client app (e.g., loading tokenizer)
    mlmodel.user_defined_metadata["co.huggingface.exporters.name"] = MODEL_ID
    logger.info("Metadata added.")

    # --- Define Output Path ---
    # Specific name indicating compression strategy
    # Output filename can remain the same or be updated if desired
    output_filename = "Llama1Stateful_Pal4_LUT8.mlpackage" # Filename for compressed model
    output_path = os.path.join(os.path.dirname(__file__) or ".", output_filename) # Place it relative to the script
    logger.info(f"Target output path for compressed model: {output_path}")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # --- Compression: 4-bit Palettization ---
    logger.info("Starting compression: 4-bit Palettization (mode='kmeans', granularity='per_grouped_channel')...")
    try:
        # Configure palettization: 4-bit, k-means clustering, grouped channel-wise
        # These settings seemed to work for Mistral, good starting point.
        palettize_op_config = ct.optimize.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            granularity="per_grouped_channel", # Good balance for transformers
            group_size=16, # Common group size
            # weight_threshold=512 # Optionally skip small weights
        )
        palettize_config = ct.optimize.coreml.OptimizationConfig(global_config=palettize_op_config)

        # Apply palettization
        palettized_mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=palettize_config)
        logger.info("4-bit Palettization complete.")
    except Exception as e:
        logger.error(f"4-bit Palettization failed: {e}", exc_info=True)
        # If compression fails, maybe save the uncompressed validated model?
        # We already saved the FP16 model above, so just log and exit here.
        logger.error("Exiting due to palettization failure.")
        sys.exit(1) # Still exit as compression failed

    # --- Save the Palettized model (Intermediate Step) ---
    pal4_filename = "Llama1Stateful_Pal4_FP16LUT.mlpackage"
    pal4_output_path = os.path.join(os.path.dirname(__file__) or ".", pal4_filename)
    logger.info(f"Saving Palettized (FP16 LUT) Core ML model to {pal4_output_path}...")
    try:
        palettized_mlmodel.save(pal4_output_path)
        logger.info(f"Successfully saved Pal4 model to {pal4_output_path}")
    except Exception as e:
        logger.error(f"Failed to save Pal4 model: {e}", exc_info=True)
        # Continue to joint quantization anyway, but log the failure
        logger.warning("Continuing with joint quantization despite Pal4 save failure.")


    # --- Compression: 8-bit LUT Linear Quantization (Jointly Applied) ---
    logger.info("Starting joint compression: 8-bit LUT Linear Quantization (mode='linear_symmetric', granularity='per_tensor')...")
    try:
        # Configure LUT quantization: linear symmetric, INT8 for LUT values.
        # Granularity MUST be 'per_tensor' for joint LUT quantization.
        lut_quant_op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8", # Quantize the LUT itself to INT8
            granularity="per_tensor", # Required for joint LUT quantization
        )
        lut_quant_config = ct.optimize.coreml.OptimizationConfig(global_config=lut_quant_op_config)

        # Apply quantization jointly with the existing palettization
        joint_compressed_mlmodel = ct.optimize.coreml.linear_quantize_weights(
            palettized_mlmodel, # Apply to the already palettized model
            config=lut_quant_config,
            joint_compression=True # Critical flag
        )
        logger.info("8-bit LUT Linear Quantization complete (jointly applied).")
    except Exception as e:
        logger.error(f"8-bit LUT Linear Quantization failed: {e}", exc_info=True)
        # If joint compression fails, maybe save the palettized model?
        # We already attempted to save the palettized model above.
        logger.error("Exiting due to joint LUT quantization failure.")
        sys.exit(1) # Exit as full compression failed


    # --- Save the Final Compressed Model ---
    logger.info(f"Saving final COMPRESSED Core ML model to {output_path}...")
    try:
        # Save the final compressed model object
        joint_compressed_mlmodel.save(output_path) # Save the compressed model
        logger.info(f"Successfully saved compressed model to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
        logger.info("Script finished successfully.")
    except Exception:
        logger.error("Unhandled exception during conversion:", exc_info=True)
        sys.exit(1)
# Note: Pruning is not included in this script, focusing on palettization + LUT quantization first.

# --- Example SCP commands (keep or remove as needed) ---
# scp -i aws-mac-key.pem stateful_convert_llama3_to_coreml.py \
#     ec2-user@54.196.141.124:/Users/ec2-user/llama3-conversion

# scp -i aws-mac-key.pem -r ec2-user@54.196.141.124:REDO_Llama3Stateful_Pal4_LUT8.mlpackage \
#     ./

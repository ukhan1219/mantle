import os
import sys
import logging
import traceback
import torch
import torch.nn as nn
from transformers import MistralForCausalLM
import coremltools as ct
import numpy as np

# ──── Logging Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("mistral_conversion")
logger.setLevel(logging.DEBUG)

fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
formatter = logging.Formatter(fmt)

# Console handler for high-level info
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler for detailed debug
fh = logging.FileHandler("conversion.log", mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# ──── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.2"
CONTEXT_LENGTH = 2048
BATCH_SIZE     = 1

# Set up device and dtype
USE_CPU = False  # Try MPS first
if not USE_CPU and torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS device for conversion")
else:
    device = torch.device("cpu")
    USE_CPU = True
    logger.info("MPS not available or USE_CPU=True, falling back to CPU.")

dtype = torch.float16
logger.info(f"Using device: {device}, dtype: {dtype}")

class StatefulMistralWrapper(nn.Module):
    """
    Wrapper around MistralForCausalLM that manages KV cache state via registered buffers.
    """
    def __init__(self, model_id: str, batch_size: int = 1, context_length: int = 2048):
        super().__init__()
        logger.debug("Loading MistralForCausalLM model...")
        self.model = MistralForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        ).to(device).eval()
        self.config = self.model.config

        # Compute KV-cache shape
        head_dim     = self.config.hidden_size // self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        self.kv_cache_shape = (
            batch_size,
            self.config.num_hidden_layers,
            num_kv_heads,
            context_length,
            head_dim
        )
        logger.debug(f"KV cache shape: {self.kv_cache_shape}")

        # Register KV cache buffers
        self.register_buffer(
            "keyCache",
            torch.zeros(self.kv_cache_shape, dtype=dtype, device=device),
            persistent=False
        )
        self.register_buffer(
            "valueCache",
            torch.zeros(self.kv_cache_shape, dtype=dtype, device=device),
            persistent=False
        )
        logger.info("Registered KV cache buffers.")

    def forward(self, input_ids, causal_mask):
        logger.debug("Running forward pass with use_cache=True")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            use_cache=True,
            return_dict=True
        )
        
        # Stack per-layer key/value tensors
        past_kvs = outputs.past_key_values
        keys   = torch.stack([kv[0] for kv in past_kvs], dim=1)
        values = torch.stack([kv[1] for kv in past_kvs], dim=1)

        seq_len = keys.shape[3]
        self.keyCache[:, :, :, :seq_len, :]   = keys
        self.valueCache[:, :, :, :seq_len, :] = values
        logger.debug(f"Updated KV caches for seq_len={seq_len}")

        return outputs.logits


def main():
    logger.info("Starting Mistral conversion process...")
    model = StatefulMistralWrapper(MODEL_ID, BATCH_SIZE, CONTEXT_LENGTH).eval()

    # Example inputs for tracing
    example_seq_len      = 2
    example_end_step_dim = 5
    example_input_ids    = torch.zeros((BATCH_SIZE, example_seq_len), dtype=torch.int32, device=device)
    example_causal_mask  = torch.zeros(
        (BATCH_SIZE, 1, example_seq_len, example_end_step_dim),
        dtype=dtype, device=device
    )
    logger.info("Tracing the model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (example_input_ids, example_causal_mask),
            check_trace=False
        )
    logger.info("Tracing complete.")

    # Core ML conversion setup
    query_length = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)

    inputs = [
        ct.TensorType(name="inputIds",   shape=(BATCH_SIZE, query_length), dtype=np.int32),
        ct.TensorType(name="causalMask", shape=(BATCH_SIZE, 1, query_length, end_step_dim), dtype=np.float16),
    ]
    outputs = [ ct.TensorType(name="logits", dtype=np.float16) ]
    states  = [
        ct.StateType(
            name="keyCache",
            wrapped_type=ct.TensorType(shape=model.kv_cache_shape, dtype=np.float16)
        ),
        ct.StateType(
            name="valueCache",
            wrapped_type=ct.TensorType(shape=model.kv_cache_shape, dtype=np.float16)
        )
    ]

    logger.info("Converting to Core ML (skipping model load)...")
    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        skip_model_load=True,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL
    )
    logger.info("Conversion call complete.")

    # Add metadata
    mlmodel.author  = "Hugging Face & Core ML Tools"
    mlmodel.license = "Apache 2.0"
    mlmodel.version = "1.0"
    mlmodel.short_description = (
        f"Stateful Core ML conversion of {MODEL_ID} with FP16 precision and {CONTEXT_LENGTH}-token context."
    )
    if mlmodel.is_package:
        mlmodel.user_defined_metadata["co.huggingface.exporters.name"] = MODEL_ID
    else:
        mlmodel._spec.description.metadata.userDefined["co.huggingface.exporters.name"] = MODEL_ID
    logger.info("Metadata added to model.")

    # Define the output path
    output_path = "MistralStateful_Pal4_LUT8.mlpackage"
    
    logger.info("Starting compression: 4-bit Palettization...")
    
    # --- 4-bit Palettization ---
    palettize_op_config = ct.optimize.coreml.OpPalettizerConfig(
        nbits=4,
        mode="kmeans",
        granularity="per_grouped_channel",
        group_size=16,
    )
    palettize_config = ct.optimize.coreml.OptimizationConfig(global_config=palettize_op_config)
    
    palettized_mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=palettize_config)
    logger.info("4-bit Palettization complete.")
    
    logger.info("Starting compression: 8-bit LUT Linear Quantization...")
    # --- 8-bit LUT Linear Quantization (Joint Compression) ---
    # Note: granularity must be 'per_tensor' for LUT quantization in joint compression
    lut_quant_op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric", 
        dtype="int8", # Quantize LUT to INT8
        granularity="per_tensor",
    )
    lut_quant_config = ct.optimize.coreml.OptimizationConfig(global_config=lut_quant_op_config)
    
    joint_compressed_mlmodel = ct.optimize.coreml.linear_quantize_weights(
        palettized_mlmodel, 
        config=lut_quant_config, 
        joint_compression=True
    )
    logger.info("8-bit LUT Linear Quantization complete.")

    # --- Save the final compressed model ---
    logger.info(f"Saving final compressed Core ML model to {output_path}...")
    joint_compressed_mlmodel.save(output_path)
    logger.info(f"Palettization and LUT Quantization complete. Model saved to {output_path}.")


if __name__ == "__main__":
    try:
        main()
        logger.info("Script finished without errors.")
    except Exception:
        logger.error("Unhandled exception during conversion:", exc_info=True)
        sys.exit(1)

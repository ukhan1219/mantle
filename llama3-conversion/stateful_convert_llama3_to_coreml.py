import os
import sys
import logging
import traceback
from typing import Any, Optional, Sequence # Added for SliceUpdateKeyValueCache

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoConfig # Changed import and added AutoConfig
from transformers.cache_utils import Cache # Added for SliceUpdateKeyValueCache
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
MODEL_ID       = "meta-llama/Llama-3.2-1B" # Confirmed MODEL_ID
CONTEXT_LENGTH = 2048 # Max sequence length for KV cache
BATCH_SIZE     = 1

# Set up device and dtype
USE_CPU = False  # Try MPS first
if not USE_CPU and torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS device for conversion")
elif torch.cuda.is_available(): # Check for CUDA as well
    device = torch.device("cuda")
    logger.info("Using CUDA device for conversion")
else:
    device = torch.device("cpu")
    USE_CPU = True
    logger.info("MPS/CUDA not available or USE_CPU=True, falling back to CPU.")

dtype = torch.float16
logger.info(f"Using device: {device}, dtype: {dtype}")

# --- Custom KV Cache Implementation (Inspired by Llama 3.1 Core ML Guide) ---
class SliceUpdateKeyValueCache(Cache):
    """
    Helper class for in-place slice updating key/value caches.
    Manages the tensors for keys and values, updating them via slicing.
    Shape expected: (#layers, batch_size, #kv_heads, context_size, head_dim)
    """
    def __init__(
        self,
        shape: Sequence[int],
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.past_seen_tokens: int = 0 # Tracks the number of tokens already in the cache
        # Allocate the actual tensors for K and V cache
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        logger.debug(f"Initialized SliceUpdateKeyValueCache with K/V shape: {shape}, dtype: {dtype}, device: {device}")

    def update(
        self,
        k_state: torch.Tensor, # New key states for the current input tokens
        v_state: torch.Tensor, # New value states for the current input tokens
        layer_idx: int,        # The layer index for which to update the cache
        cache_kwargs: Optional[dict[str, Any]] = None, # Contains 'cache_position'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache tensors for a specific layer (`layer_idx`)
        with the new key/value states (`k_state`, `v_state`).
        Uses `self.past_seen_tokens` to determine the correct slice for the update.
        Returns the *entire* K and V cache states up to the *new* total sequence length.
        """
        # Determine the start and end indices for the update slice
        # `k_state.shape[1]` (num_kv_heads) might be smaller if GQA is used, adjust slice accordingly
        new_seq_len = k_state.shape[-2] # Sequence length of the new states being added
        start_pos = self.past_seen_tokens
        end_pos = self.past_seen_tokens + new_seq_len
        logger.debug(f"Updating Layer {layer_idx}: start={start_pos}, end={end_pos}, new_len={new_seq_len}")

        # Update the cache tensors using slicing
        self.k[layer_idx, :, : k_state.shape[1], start_pos:end_pos, :] = k_state
        self.v[layer_idx, :, : v_state.shape[1], start_pos:end_pos, :] = v_state

        # Return the slices of the cache containing all tokens up to the new length (`end_pos`)
        # These are the K and V values the attention mechanism should use
        # The shape needs to match what the scaled_dot_product_attention expects:
        # (batch_size, num_kv_heads, sequence_length, head_dim)
        # Our cache stores K/V as (num_layers, batch, kv_heads, context, head_dim)
        # Slicing the layer dim gives: (batch, kv_heads, context, head_dim)
        # We need slice up to end_pos: (batch, kv_heads, end_pos, head_dim)
        updated_k = self.k[layer_idx, :, :, :end_pos, :]
        updated_v = self.v[layer_idx, :, :, :end_pos, :]
        logger.debug(f"Layer {layer_idx} updated K shape: {updated_k.shape}, V shape: {updated_v.shape}")
        return updated_k, updated_v

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the number of tokens currently stored in the cache."""
        # This is called by the model internally, reflects the state *before* the current update.
        return self.past_seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length capacity of this cache."""
        return self.k.shape[-2] # context_length dimension

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, required by Cache interface."""
        # Not strictly needed for greedy search conversion, but good practice to implement
        self.k = self.k.index_select(1, beam_idx)
        self.v = self.v.index_select(1, beam_idx)
        logger.warning("SliceUpdateKeyValueCache reorder_cache called (not optimized for beam search).")


# --- Refactored Stateful Wrapper (Uses SliceUpdateKeyValueCache) ---
class StatefulLlamaWrapper(nn.Module): # Renamed class reflects Llama 3
    """
    Wrapper around LlamaForCausalLM that manages KV cache state via a custom Cache object
    and registered buffers, suitable for Core ML stateful conversion.
    """
    def __init__(self, model_id: str, batch_size: int = BATCH_SIZE, context_length: int = CONTEXT_LENGTH, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        logger.debug(f"Loading LlamaForCausalLM model: {model_id}...")
        # Load config first to get parameters like num_key_value_heads
        self.config = AutoConfig.from_pretrained(model_id)
        logger.info(f"Loaded model config for {model_id}:")
        logger.info(f"  - num_hidden_layers: {self.config.num_hidden_layers}")
        logger.info(f"  - num_attention_heads: {self.config.num_attention_heads}")
        logger.info(f"  - num_key_value_heads: {self.config.num_key_value_heads}")
        logger.info(f"  - hidden_size: {self.config.hidden_size}")
        logger.info(f"  - vocab_size: {self.config.vocab_size}") # Log vocab size for validation

        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            config=self.config, # Pass loaded config
            torch_dtype=dtype,
            # device_map="auto", # Avoid device_map for single device setup
            low_cpu_mem_usage=True, # Keep low_cpu_mem_usage
            attn_implementation="eager" # Use eager attention for tracing & SDPA fusion
        ).to(device).eval()
        logger.info(f"Model {model_id} loaded successfully onto {device}.")

        # --- Compute KV-cache shape using Llama 3 config attributes ---
        num_kv_heads = self.config.num_key_value_heads
        head_dim     = self.config.hidden_size // self.config.num_attention_heads
        num_layers   = self.config.num_hidden_layers
        self.vocab_size = self.config.vocab_size # Store vocab size

        if not num_kv_heads:
            logger.warning("`num_key_value_heads` not found in config, assuming equal to `num_attention_heads`.")
            num_kv_heads = self.config.num_attention_heads

        # Shape for the *state buffer* which holds cache for all layers
        # Format: (num_layers, batch_size, num_kv_heads, max_sequence_length, head_dim) - Note layer dim first for SliceUpdateKeyValueCache
        self.kv_cache_shape = (
            num_layers,
            batch_size,
            num_kv_heads,
            context_length, # Use the maximum context length for the buffer
            head_dim
        )
        logger.info(f"KV cache state buffer shape computed: {self.kv_cache_shape}")
        logger.info(f" (Layers={num_layers}, Batch={batch_size}, KV_Heads={num_kv_heads}, Context={context_length}, HeadDim={head_dim})")

        # --- Instantiate the Custom Cache ---
        # The K/V tensors inside this cache object will be registered as buffers.
        self.kv_cache = SliceUpdateKeyValueCache(
            shape=self.kv_cache_shape, dtype=dtype, device=device
        )

        # --- Register KV cache buffers FROM the custom cache instance ---
        # These buffers will hold the state across inference calls in the Core ML model.
        # `persistent=False` means they are part of the state but not saved in state_dict.
        # Names MUST match the `states` definition in `ct.convert`.
        self.register_buffer(
            "keyCache",
            self.kv_cache.k, # Register the tensor from the cache object
            persistent=False
        )
        self.register_buffer(
            "valueCache",
            self.kv_cache.v, # Register the tensor from the cache object
            persistent=False
        )
        logger.info("Registered KV cache buffers ('keyCache', 'valueCache') from SliceUpdateKeyValueCache.")


    def forward(self, input_ids, causal_mask):
        # input_ids shape: (batch_size, sequence_length)
        # causal_mask shape: (batch_size, 1, query_length, key_value_sequence_length)
        logger.debug(f"Wrapper forward called.")
        logger.debug(f"  Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        logger.debug(f"  Causal Mask shape: {causal_mask.shape}, dtype: {causal_mask.dtype}")

        # Determine the number of tokens already processed (past sequence length)
        # This is crucial for the SliceUpdateKeyValueCache to know where to start updating.
        # For tracing, the mask's last dimension (`key_value_sequence_length`) should reflect the total sequence length
        # including the past kv cache length. The input_ids length is the new tokens being added.
        current_input_seq_len = input_ids.shape[-1]
        total_seq_len_incl_past = causal_mask.shape[-1]
        self.kv_cache.past_seen_tokens = total_seq_len_incl_past - current_input_seq_len
        logger.debug(f"  Calculated past_seen_tokens: {self.kv_cache.past_seen_tokens} (Total: {total_seq_len_incl_past}, Current: {current_input_seq_len})")

        # Run the underlying Llama model, passing the custom cache object
        # `use_cache=True` ensures the model attempts to use and update the cache.
        # The model will internally call self.kv_cache.update() for each layer.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=causal_mask, # Pass the 4D float mask directly
            past_key_values=self.kv_cache, # Pass the custom cache object
            use_cache=True,
            return_dict=True
        )
        logger.debug(f"  Model forward pass completed. Output type: {type(outputs)}")

        # --- KV Cache State is updated *implicitly* by the model call via SliceUpdateKeyValueCache.update ---
        # No need to manually extract past_key_values and update buffers here.

        if not hasattr(outputs, 'logits'):
             logger.error("Model output does not contain 'logits'.")
             # Handle error appropriately, maybe raise or return None
             return None # Or raise an exception

        logger.debug(f"  Returning logits shape: {outputs.logits.shape}")
        # Return only the logits, as the state update happened implicitly via the cache object
        # modifying the registered buffers.
        return outputs.logits


# --- Implemented Core ML Validation Function ---
def validate_coreml_model(mlmodel: ct.models.MLModel, batch_size: int, vocab_size: int, device: torch.device):
    """ Perform a basic prediction using the Core ML model to validate conversion. """
    logger.info("--- Starting Core ML Model Validation ---")
    try:
        # Prepare dummy input data based on the model's expected input spec
        # Use default dimensions from RangeDim (default=1) for a simple test
        default_seq_len = 1
        input_ids_shape = (batch_size, default_seq_len)
        # For the mask, query_length=1, key_value_length=1 (initial state)
        causal_mask_shape = (batch_size, 1, default_seq_len, default_seq_len) # Query=1, Key=1

        logger.debug(f"Validation input shapes: inputIds={input_ids_shape}, causalMask={causal_mask_shape}")

        # Create dummy NumPy inputs
        # Input IDs should be within vocab size, but 0 is fine for shape testing
        input_ids_np = np.zeros(input_ids_shape, dtype=np.int32)
        # Causal mask needs to be float16, all zeros is fine for first token prediction
        causal_mask_np = np.zeros(causal_mask_shape, dtype=np.float16)

        coreml_input_dict = {
            "inputIds": input_ids_np,
            "causalMask": causal_mask_np
        }
        logger.debug(f"Validation input dict prepared: {coreml_input_dict.keys()}")

        # Initialize state using the model object
        logger.debug("Creating initial Core ML state using mlmodel.make_state()...")
        initial_state = mlmodel.make_state()
        logger.debug("Initial state created successfully.")

        # Run prediction using the initial state
        logger.debug("Running Core ML prediction for validation...")
        # Use the compute units specified during conversion/loading if possible
        # Assuming mlmodel object was created with appropriate compute_units
        output_dict = mlmodel.predict(coreml_input_dict, state=initial_state)
        logger.debug(f"Core ML prediction successful. Output keys: {output_dict.keys()}")

        # Check output presence and shape
        if "logits" not in output_dict:
            logger.error("Validation Failed: 'logits' key not found in prediction output.")
            return False

        logits_output = output_dict["logits"]
        # Expected logits shape: (batch_size, sequence_length, vocab_size)
        # For this validation: (batch_size, 1, vocab_size)
        expected_logits_shape = (batch_size, default_seq_len, vocab_size)
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
        # Use traceback to get more detailed error information
        logger.error(f"Error during validation prediction: {e}\n{traceback.format_exc()}")
        return False

# --- Main Conversion Script ---
def main():
    logger.info(f"Starting Llama 3.2 ({MODEL_ID}) stateful conversion process...")
    # Pass device and dtype to the wrapper
    model_wrapper = StatefulLlamaWrapper(MODEL_ID, BATCH_SIZE, CONTEXT_LENGTH, device=device, dtype=dtype).eval()

    # --- Tracing Inputs (Refined) ---
    # Use a small sequence length for tracing efficiency.
    example_seq_len      = 2
    # For tracing, the mask's key/value length dimension should reflect the total sequence length *at that trace step*.
    # Example: If tracing with seq_len=2, and assuming it's the *first* call (past_len=0),
    # the relevant key/value history length is also 2. If we were tracing a second step (past_len=2, new_len=1),
    # the key/value dim would be 3.
    # Let's trace the initial step (past_len=0, new_len=example_seq_len).
    # So, the 4th dim of the mask should be example_seq_len.
    example_kv_len_for_trace = example_seq_len # Total sequence length at this trace point
    logger.info(f"Tracing with: example_seq_len={example_seq_len}, causal mask kv_len={example_kv_len_for_trace}")

    example_input_ids    = torch.zeros((BATCH_SIZE, example_seq_len), dtype=torch.int32, device=device)
    # Causal mask shape: (batch_size, 1, query_length, key_value_length)
    # Use the calculated example_kv_len_for_trace for the last dimension here.
    example_causal_mask  = torch.zeros(
        (BATCH_SIZE, 1, example_seq_len, example_kv_len_for_trace), # Use example_kv_len_for_trace
        dtype=dtype, device=device # Ensure mask dtype matches model's float type
    )
    logger.info(f"Example input_ids shape: {example_input_ids.shape}, dtype: {example_input_ids.dtype}")
    logger.info(f"Example causal_mask shape: {example_causal_mask.shape}, dtype: {example_causal_mask.dtype}")

    logger.info("Tracing the model with torch.jit.trace...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model_wrapper,
                (example_input_ids, example_causal_mask),
                check_trace=False # Often needed for complex models
            )
        logger.info("Tracing complete.")
    except Exception as e:
         logger.error(f"Tracing failed: {e}", exc_info=True)
         # Try to print graph inputs/outputs on failure
         try:
             logger.error(f"Traced model graph inputs: {[inp.debugName() for inp in traced_model.graph.inputs()]}")
         except Exception as ge:
              logger.error(f"Could not get graph inputs: {ge}")
         sys.exit(1)


    # --- Core ML Conversion Setup ---
    # Define flexible input dimensions using RangeDim
    # Query length can vary from 1 up to CONTEXT_LENGTH
    query_length = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)
    # Key/value length dimension in the mask also needs to be flexible, representing total context.
    key_value_length_dim = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)
    logger.debug(f"Defining Core ML input shapes with query_length=(1,{CONTEXT_LENGTH}) and key_value_length_dim=(1,{CONTEXT_LENGTH})")

    # Define Core ML input tensor types
    inputs = [
        ct.TensorType(name="inputIds",   shape=(BATCH_SIZE, query_length), dtype=np.int32),
        # Mask needs to be Float16 as per Llama 3.1 example and common practice for fused SDPA
        ct.TensorType(name="causalMask", shape=(BATCH_SIZE, 1, query_length, key_value_length_dim), dtype=np.float16),
    ]
    # Define Core ML output tensor types
    outputs = [ ct.TensorType(name="logits", dtype=np.float16) ] # Match model's float type
    # Define Core ML state types, names must match registered buffers ("keyCache", "valueCache")
    states  = [
        ct.StateType(
            name="keyCache", # Must match buffer name in wrapper
            wrapped_type=ct.TensorType(shape=model_wrapper.kv_cache_shape, dtype=np.float16) # Match model's float type 
            
        ),
        ct.StateType(
            name="valueCache", # Must match buffer name in wrapper
            wrapped_type=ct.TensorType(shape=model_wrapper.kv_cache_shape, dtype=np.float16) # Match model's float type
        )
    ]
    logger.info("Core ML input, output, and state types defined.")


    # --- Core ML Conversion ---
    # Convert to ML Program format for stateful models (iOS 18+ / macOS 15+)
    logger.info("Converting traced model to Core ML ML Program (FP16)...")
    # Set compute units for conversion - ALL allows NE/GPU/CPU usage
    # Using Float16 precision for computation initially
    # Ensure minimum target is set for stateful and SDPA features
    try:
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            states=states,
            # convert_to="mlprogram", # Implicit with iOS18/macOS15 target
            # skip_model_load=False, # Keep False to allow validation on the loaded object
            minimum_deployment_target=ct.target.iOS18, # Required for stateful models & encourages SDPA
            compute_precision=ct.precision.FLOAT16, # Use FP16 for weights/activations initially
            compute_units=ct.ComputeUnit.ALL # Allow ANE, GPU, CPU
        )
        logger.info("Core ML conversion call successful.")
    except Exception as e:
         logger.error(f"Core ML conversion failed: {e}", exc_info=True)
         # Print model input/output signature if available for debugging
         try:
             logger.error(f"Traced model signature: Inputs: {[inp.debugName() for inp in traced_model.graph.inputs()]}")
             # logger.error(f"Traced model signature: Outputs: {traced_model.graph.outputs()}") # Outputs might be complex
         except Exception as sig_e:
             logger.error(f"Could not get traced model signature: {sig_e}")
         sys.exit(1)

    # --- Inline Validation ---
    # Pass necessary info like vocab_size from the wrapper
    if not validate_coreml_model(mlmodel, BATCH_SIZE, model_wrapper.vocab_size, device):
        logger.error("Model validation failed. Exiting before compression/saving.")
        sys.exit(1)
    # If validation passed, we can proceed.

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
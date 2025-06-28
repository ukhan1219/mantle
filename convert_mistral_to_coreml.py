import os
import torch
import torch.nn as nn
from transformers import MistralForCausalLM, AutoTokenizer
import coremltools as ct
import numpy as np

# Configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
CONTEXT_LENGTH = 2048
BATCH_SIZE = 1

# Set up device and dtype
# Use CPU for potentially more stable conversion, or MPS if available/preferred
USE_CPU = True # Force CPU
# USE_CPU = False  # Try MPS first
if not USE_CPU and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    USE_CPU = True # Fallback to CPU if MPS not available
    print("MPS not available or USE_CPU=True, falling back to CPU.")
    
dtype = torch.float16
print(f"Using device: {device}, dtype: {dtype}")

class StatefulMistralWrapper(nn.Module):
    """
    A wrapper around MistralForCausalLM that manages KV cache state.
    This version avoids registering buffers directly for the KV cache to work around
    state handling issues in Core ML conversion.
    """
    def __init__(self, model_id: str, batch_size: int = 1, context_length: int = 2048):
        super().__init__()
        
        # Load the base model
        self.model = MistralForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        self.config = self.model.config
        
        # Ensure model is on the target device and in eval mode
        self.model = self.model.to(device) 
        self.model.eval()

        # Calculate KV cache shapes based on Mistral's GQA architecture
        head_dim = self.config.hidden_size // self.config.num_attention_heads  # 4096/32 = 128
        num_kv_heads = self.config.num_key_value_heads  # 8 for Mistral
        
        self.kv_cache_shape = (
            batch_size,
            self.config.num_hidden_layers,
            num_kv_heads,
            context_length,
            head_dim
        )
        
        # Important: Instead of registering buffers, we'll handle state extraction differently
        # during the ct.convert process
    
    def forward(self, input_ids, causal_mask, key_cache=None, value_cache=None):
        """
        Modified forward pass that explicitly accepts key_cache and value_cache as inputs.
        This allows the CoreML converter to treat them as regular inputs rather than states.
        """
        # If key_cache and value_cache are not provided, create zero tensors
        if key_cache is None:
            key_cache = torch.zeros(self.kv_cache_shape, dtype=dtype, device=device)
        if value_cache is None:
            value_cache = torch.zeros(self.kv_cache_shape, dtype=dtype, device=device)
            
        # Store the key and value caches in the model's past_key_values format
        # For Mistral, this is a tuple of tuples: (layer_idx, (key, value))
        past_key_values = []
        for layer_idx in range(self.config.num_hidden_layers):
            layer_key_cache = key_cache[:, layer_idx]
            layer_value_cache = value_cache[:, layer_idx]
            past_key_values.append((layer_key_cache, layer_value_cache))
            
        # Forward pass with explicit past_key_values
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        
        # Extract updated key and value caches
        new_key_cache = torch.zeros_like(key_cache)
        new_value_cache = torch.zeros_like(value_cache)
        
        for layer_idx, (key, value) in enumerate(outputs.past_key_values):
            new_key_cache[:, layer_idx] = key
            new_value_cache[:, layer_idx] = value
            
        # Return logits and updated caches
        return outputs.logits, new_key_cache, new_value_cache

def main():
    print("Starting Mistral conversion process...")
    
    # Initialize the wrapper
    print("Initializing StatefulMistral wrapper...")
    model = StatefulMistralWrapper(MODEL_ID, BATCH_SIZE, CONTEXT_LENGTH)
    model.eval()
    
    # Prepare example inputs for tracing
    print("Preparing example inputs for tracing...")
    example_seq_len = 2
    example_end_step_dim = 5
    example_input_ids = torch.zeros((BATCH_SIZE, example_seq_len), dtype=torch.int32, device=device)
    example_causal_mask = torch.zeros((BATCH_SIZE, 1, example_seq_len, example_end_step_dim), dtype=dtype, device=device)
    example_key_cache = torch.zeros(model.kv_cache_shape, dtype=dtype, device=device)
    example_value_cache = torch.zeros(model.kv_cache_shape, dtype=dtype, device=device)

    print("Input shapes:", example_input_ids.shape, example_causal_mask.shape)
    print("Cache shapes:", example_key_cache.shape, example_value_cache.shape)
    print("Input dtypes:", example_input_ids.dtype, example_causal_mask.dtype)
    print("Cache dtypes:", example_key_cache.dtype, example_value_cache.dtype)
    print("Device:", device)

    # Trace the model with explicit cache tensors
    print("Tracing the model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (example_input_ids, example_causal_mask, example_key_cache, example_value_cache),
            check_trace=False
        )

    # Define inputs
    print("Defining Core ML inputs and outputs...")
    query_length = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)

    # Now define all inputs, including the caches
    inputs = [
        ct.TensorType(
            name="inputIds",
            shape=(BATCH_SIZE, query_length),
            dtype=np.int32
        ),
        ct.TensorType(
            name="causalMask",
            shape=(BATCH_SIZE, 1, query_length, end_step_dim),
            dtype=np.float16
        ),
        ct.TensorType(
            name="keyCache",
            shape=model.kv_cache_shape,
            dtype=np.float16
        ),
        ct.TensorType(
            name="valueCache",
            shape=model.kv_cache_shape,
            dtype=np.float16
        )
    ]

    # Define outputs - include the updated caches
    outputs = [
        ct.TensorType(name="logits", dtype=np.float16),
        ct.TensorType(name="new_keyCache", shape=model.kv_cache_shape, dtype=np.float16),
        ct.TensorType(name="new_valueCache", shape=model.kv_cache_shape, dtype=np.float16)
    ]
    
    # Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16
    )
    
    # Add metadata
    print("Adding metadata...")
    mlmodel.author = "Hugging Face & Core ML Tools"
    mlmodel.license = "Apache 2.0"
    mlmodel.version = "1.0"
    mlmodel.short_description = (
        f"Core ML conversion of {MODEL_ID} with FP16 precision "
        f"and {CONTEXT_LENGTH} context length."
    )
    # Add the crucial Hugging Face model ID for tokenizer loading later
    if mlmodel.is_package:
         mlmodel.user_defined_metadata["co.huggingface.exporters.name"] = MODEL_ID
    else:
        mlmodel._spec.description.metadata.userDefined["co.huggingface.exporters.name"] = MODEL_ID


    # Save the model
    output_path = "MistralInstructFP16.mlpackage"
    print(f"Saving Core ML model to {output_path}...")
    mlmodel.save(output_path)
    print("Conversion complete!")

if __name__ == "__main__":
    main() 
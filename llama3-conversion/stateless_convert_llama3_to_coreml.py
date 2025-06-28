import os

import sys

import logging

import traceback

import torch

import torch.nn as nn

from transformers import LlamaForCausalLM # Changed import

import coremltools as ct

import numpy as np

from transformers.cache_utils import DynamicCache

  

# ──── Logging Setup ─────────────────────────────────────────────────────────────

logger = logging.getLogger("llama3_conversion") # Changed logger name

logger.setLevel(logging.DEBUG)

  

fmt = "%(asctime)s | %(levelname)-8s | %(message)s"

formatter = logging.Formatter(fmt)

  

# Console handler for high-level info

ch = logging.StreamHandler(sys.stdout)

ch.setLevel(logging.INFO)

ch.setFormatter(formatter)

logger.addHandler(ch)

  

# File handler for detailed debug

fh = logging.FileHandler("llama3_conversion.log", mode="w", encoding="utf-8") # Changed log file name

fh.setLevel(logging.DEBUG)

fh.setFormatter(formatter)

logger.addHandler(fh)

  

# ──── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "meta-llama/Llama-3.2-3B" # Changed MODEL_ID

CONTEXT_LENGTH = 2048 # Changed CONTEXT_LENGTH to 8k

BATCH_SIZE = 1

  

# Set up device and dtype

USE_CPU = False # Try MPS first

if not USE_CPU and torch.backends.mps.is_available():

device = torch.device("mps")

logger.info("Using MPS device for conversion")

else:

device = torch.device("cpu")

USE_CPU = True

logger.info("MPS not available or USE_CPU=True, falling back to CPU.")

  

dtype = torch.float16

logger.info(f"Using device: {device}, dtype: {dtype}")

  

# NEW Wrapper for Stateless Conversion

class StatelessLlamaWrapper(nn.Module):

"""Wraps LlamaForCausalLM to extract tensors for JIT tracing."""

def __init__(self, model):

super().__init__()

self.model = model

self.config = model.config

  

# Modified forward to accept past_kv tensors directly

def forward(self, input_ids, attention_mask, *past_kv_inputs):

# *past_kv_inputs will be a flat tuple: (past_key_0, past_value_0, past_key_1, past_value_1, ...)

# We need to reconstruct the nested tuple structure and potentially convert to a Cache object

past_cache_to_pass = None # Initialize as None

if len(past_kv_inputs) > 0:

if len(past_kv_inputs) % 2 != 0:

raise ValueError("Expected an even number of past_key_value tensors")

num_past_layers = len(past_kv_inputs) // 2

# Ensure the number of *input* KV pairs matches the model's layers

if num_past_layers != self.model.config.num_hidden_layers:

# This might happen if tracing with dummy inputs of wrong length or structure

logger.warning(f"Received {num_past_layers} past KV pairs, but model expects {self.model.config.num_hidden_layers}. Check trace inputs.")

# Attempt to proceed, might fail later

# Reconstruct the tuple format expected by from_legacy_cache

past_key_values_tuple = tuple(past_kv_inputs[i:i+2] for i in range(0, len(past_kv_inputs), 2))

# Convert the tuple to a DynamicCache object

try:

past_cache_to_pass = DynamicCache.from_legacy_cache(past_key_values=past_key_values_tuple)

logger.debug("Successfully created DynamicCache from past_kv_inputs.")

except Exception as e:

logger.error(f"Failed to create DynamicCache from legacy tuple: {e}", exc_info=True)

raise e # Re-raise after logging

  

outputs = self.model(

input_ids=input_ids,

attention_mask=attention_mask,

past_key_values=past_cache_to_pass, # Pass the Cache object or None

use_cache=True,

return_dict=True

)

  

# Prepare outputs for JIT trace (only tensors in dict/tuple)

# traced_outputs = {"logits": outputs.logits}

# Instead of dict, use a list, then convert to tuple

output_tensors = [outputs.logits]

  

# Extract K/V tensors from the cache object

# outputs.past_key_values is now a Cache object (e.g., DynamicCache)

if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:

# Access tensors from the cache object's attributes

key_states = outputs.past_key_values.key_cache

value_states = outputs.past_key_values.value_cache

if len(key_states) != len(value_states):

logger.error(f"Mismatch between key ({len(key_states)}) and value ({len(value_states)}) states in cache object")

# Decide how to handle error, maybe raise?

elif len(key_states) != self.config.num_hidden_layers:

logger.warning(f"Number of key states in cache ({len(key_states)}) doesn't match model layers ({self.config.num_hidden_layers})")

  

for key_tensor, value_tensor in zip(key_states, value_states):

output_tensors.append(key_tensor)

output_tensors.append(value_tensor)

else:

logger.warning("Could not find past_key_values in model output during trace.")

  

# Return a tuple instead of a dict

return tuple(output_tensors)

  

def main():

logger.info("Starting Llama 3.2 STATLESS conversion process...") # Updated log message

# Load base model directly

logger.info(f"Loading base LlamaForCausalLM model: {MODEL_ID}")

base_model = LlamaForCausalLM.from_pretrained( # Renamed to base_model

MODEL_ID,

torch_dtype=dtype,

device_map="auto",

low_cpu_mem_usage=True,

attn_implementation="eager" # Keep eager for compatibility during trace

).to(device).eval()

config = base_model.config # Get config from base_model

logger.info("Base model loaded.")

  

# Instantiate the stateless wrapper

model = StatelessLlamaWrapper(base_model).eval()

  

# --- Calculate expected KV cache shape PER LAYER --- Needed for defining I/O

head_dim = config.hidden_size // config.num_attention_heads

num_kv_heads = config.num_key_value_heads

num_layers = config.num_hidden_layers

# Shape for ONE layer's key/value cache: (batch, num_kv_heads, seq_len, head_dim)

# We use CONTEXT_LENGTH for the seq_len dimension in the definition

kv_shape_per_layer = (

BATCH_SIZE,

num_kv_heads,

CONTEXT_LENGTH,

head_dim

)

logger.info(f"KV cache shape per layer: {kv_shape_per_layer}")

  

# Example inputs for tracing the BASE model

example_seq_len = 2 # Keep small

example_input_ids = torch.zeros((BATCH_SIZE, example_seq_len), dtype=torch.int32, device=device)

# Base model forward might only need 2D mask, but let's try with 4D causal mask first

# If tracing fails, try a 2D mask: torch.ones(...) instead

example_causal_mask = torch.ones(

(BATCH_SIZE, 1, example_seq_len, example_seq_len), # Q=K=example_seq_len for base trace

dtype=dtype,

device=device

)

# Example past_key_values for tracing (Tuple of tuples (key, value) per layer)

# Create dummy past KV tensors for the trace call

# Shape: B, num_kv_heads, PAST_seq_len=0, head_dim

example_past_kv_seq_len = 0 # Match lower bound of past_kv_seq_len_dim

example_past_kv_shape = (

BATCH_SIZE, num_kv_heads, example_past_kv_seq_len, head_dim

)

dummy_past_kv_tensors = []

for _ in range(num_layers):

dummy_key = torch.zeros(example_past_kv_shape, dtype=dtype, device=device)

dummy_value = torch.zeros(example_past_kv_shape, dtype=dtype, device=device)

dummy_past_kv_tensors.extend([dummy_key, dummy_value])

dummy_past_kv_tensors = tuple(dummy_past_kv_tensors)

logger.info(f"Created {len(dummy_past_kv_tensors)} dummy past KV tensors for tracing with shape {example_past_kv_shape}")

  

# Define the full list of example inputs for the trace

example_trace_inputs = (example_input_ids, example_causal_mask) + dummy_past_kv_tensors

  

logger.info(f"Tracing the StatelessLlamaWrapper... Total inputs: {len(example_trace_inputs)}")

  

# Trace the WRAPPER model.

with torch.no_grad():

# Pass the full tuple including dummy past KVs

traced_model = torch.jit.trace(

model,

example_trace_inputs, # Pass the combined inputs

check_trace=False

)

logger.info("Tracing complete.")

  

# Core ML conversion setup - STATELESS

query_length = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)

# Full context dim for mask K/V length

key_value_length_dim = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)

  

# --- Define Inputs ---

inputs = [

ct.TensorType(name="inputIds", shape=(BATCH_SIZE, query_length), dtype=np.int32),

# The mask's key/value dimension should match the stateful cache's capacity

ct.TensorType(name="causalMask", shape=(BATCH_SIZE, 1, query_length, key_value_length_dim), dtype=np.float16),

]

# Add explicit PAST KV inputs for stateless model

# The sequence length dimension here should represent the PAST length

past_kv_seq_len_dim = ct.RangeDim(lower_bound=0, upper_bound=CONTEXT_LENGTH-1, default=0)

for i in range(num_layers):

inputs.append(ct.TensorType(name=f"past_key_{i}",

# Shape: B, num_kv_heads, PAST_seq_len, head_dim

shape=(BATCH_SIZE, num_kv_heads, past_kv_seq_len_dim, head_dim),

dtype=np.float16))

inputs.append(ct.TensorType(name=f"past_value_{i}",

# Shape: B, num_kv_heads, PAST_seq_len, head_dim

shape=(BATCH_SIZE, num_kv_heads, past_kv_seq_len_dim, head_dim),

dtype=np.float16))

logger.info(f"Defined {len(inputs)} total inputs for Core ML conversion.")

  

# --- Define Outputs ---

# Logits output

outputs = [ ct.TensorType(name="logits", dtype=np.float16) ]

# Add explicit PRESENT KV outputs

# The sequence length dimension here should represent the PRESENT (updated) length

# Output shape should reflect the concatenation of past + current

# Let's assume output KV cache shape matches the kv_shape_per_layer definition for simplicity

# We might need to adjust this based on how predict handles varying shapes

present_kv_seq_len_dim = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=example_seq_len)

for i in range(num_layers):

outputs.append(ct.TensorType(name=f"present_key_{i}",

# Shape is inferred by CoreML, do not specify

dtype=np.float16))

outputs.append(ct.TensorType(name=f"present_value_{i}",

# Shape is inferred by CoreML, do not specify

dtype=np.float16))

logger.info(f"Defined {len(outputs)} total outputs for Core ML conversion.")

  

logger.info("Converting to Core ML (stateless, skipping model load)...") # Updated log

mlmodel = ct.convert(

traced_model,

inputs=inputs,

outputs=outputs,

skip_model_load=True,

minimum_deployment_target=ct.target.iOS18,

compute_precision=ct.precision.FLOAT16,

compute_units=ct.ComputeUnit.ALL

)

logger.info("Conversion call complete.")

  

# Add metadata

mlmodel.author = "Meta & Core ML Tools" # Updated author

mlmodel.license = "Llama 3.2 Community License" # Updated license

mlmodel.version = "1.0"

mlmodel.short_description = ( # Updated description for stateless

f"Stateless Core ML conversion of {MODEL_ID} with FP16 precision, {CONTEXT_LENGTH}-token context, "

f"4-bit Palettization, and 8-bit LUT quantization."

)

# Use consistent metadata setting

mlmodel.user_defined_metadata["co.huggingface.exporters.name"] = MODEL_ID

logger.info("Metadata added to model.")

  

# Define the output path

# output_path = "Llama3Stateful_FP16.mlpackage" # Path for previous test

output_path = "Llama3Stateless_Pal4_LUT8.mlpackage" # New path for stateless compressed model

logger.info(f"Target output path: {output_path}")

  

# --- Re-enable Compression --- #

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

lut_quant_op_config = ct.optimize.coreml.OpLinearQuantizerConfig(

mode="linear_symmetric",

dtype="int8",

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

logger.info(f"Saving final compressed stateless Core ML model to {output_path}...")

joint_compressed_mlmodel.save(output_path)

logger.info(f"Stateless Palettization and LUT Quantization complete. Model saved to {output_path}.")

  
  

if __name__ == "__main__":

try:

main()

logger.info("Script finished without errors.")

except Exception:

logger.error("Unhandled exception during conversion:", exc_info=True)

sys.exit(1)

# # NO STATES 6

  

# import os

# import sys

# import logging

# import traceback

# import torch

# import torch.nn as nn

# from transformers import LlamaForCausalLM # Changed import

# import coremltools as ct

# import numpy as np

  

# # ──── Logging Setup ─────────────────────────────────────────────────────────────

# logger = logging.getLogger("llama3_conversion") # Changed logger name

# logger.setLevel(logging.DEBUG)

  

# fmt = "%(asctime)s | %(levelname)-8s | %(message)s"

# formatter = logging.Formatter(fmt)

  

# # Console handler for high-level info

# ch = logging.StreamHandler(sys.stdout)

# ch.setLevel(logging.INFO)

# ch.setFormatter(formatter)

# logger.addHandler(ch)

  

# # File handler for detailed debug

# fh = logging.FileHandler("llama3_conversion.log", mode="w", encoding="utf-8") # Changed log file name

# fh.setLevel(logging.DEBUG)

# fh.setFormatter(formatter)

# logger.addHandler(fh)

  

# # ──── Configuration ─────────────────────────────────────────────────────────────

# MODEL_ID = "meta-llama/Llama-3.2-3B" # Changed MODEL_ID

# CONTEXT_LENGTH = 2048 # Changed CONTEXT_LENGTH to 8k

# BATCH_SIZE = 1

  

# # Set up device and dtype

# USE_CPU = False # Try MPS first

# if not USE_CPU and torch.backends.mps.is_available():

# device = torch.device("mps")

# logger.info("Using MPS device for conversion")

# else:

# device = torch.device("cpu")

# USE_CPU = True

# logger.info("MPS not available or USE_CPU=True, falling back to CPU.")

  

# dtype = torch.float16

# logger.info(f"Using device: {device}, dtype: {dtype}")

  

# # NEW Wrapper for Stateless Conversion

# class StatelessLlamaWrapper(nn.Module):

# """Wraps LlamaForCausalLM to extract tensors for JIT tracing."""

# def __init__(self, model):

# super().__init__()

# self.model = model

# self.config = model.config

  

# # Modified forward to accept past_kv tensors directly

# def forward(self, input_ids, attention_mask, *past_kv_inputs):

# # *past_kv_inputs will be a flat tuple: (past_key_0, past_value_0, past_key_1, past_value_1, ...)

# # We need to reconstruct the nested tuple structure: ((pk0, pv0), (pk1, pv1), ...)

# past_key_values = None

# if len(past_kv_inputs) > 0:

# if len(past_kv_inputs) % 2 != 0:

# raise ValueError("Expected an even number of past_key_value tensors")

# num_past_layers = len(past_kv_inputs) // 2

# past_key_values = tuple(past_kv_inputs[i:i+2] for i in range(0, len(past_kv_inputs), 2))

# # Ensure the reconstructed length matches the model's layers

# if len(past_key_values) != self.model.config.num_hidden_layers:

# # This might happen if tracing with dummy inputs of wrong length

# logger.warning(f"Reconstructed past_key_values length ({len(past_key_values)}) doesn't match model layers ({self.model.config.num_hidden_layers}). Check trace inputs.")

# # Attempt to proceed, might fail later

  

# outputs = self.model(

# input_ids=input_ids,

# attention_mask=attention_mask,

# past_key_values=past_key_values, # Pass the reconstructed tuple

# use_cache=True,

# return_dict=True

# )

  

# # Prepare outputs for JIT trace (only tensors in dict/tuple)

# # traced_outputs = {"logits": outputs.logits}

# # Instead of dict, use a list, then convert to tuple

# output_tensors = [outputs.logits]

  

# # Extract K/V tensors from the cache object

# # The structure might vary slightly, check transformers version if error

# if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:

# for i, (key_tensor, value_tensor) in enumerate(outputs.past_key_values):

# # traced_outputs[f"present_key_{i}"] = key_tensor

# # traced_outputs[f"present_value_{i}"] = value_tensor

# output_tensors.append(key_tensor)

# output_tensors.append(value_tensor)

# else:

# logger.warning("Could not find past_key_values in model output during trace.")

  

# # Return a tuple instead of a dict

# return tuple(output_tensors)

  

# def main():

# logger.info("Starting Llama 3.2 STATLESS conversion process...") # Updated log message

# # Load base model directly

# logger.info(f"Loading base LlamaForCausalLM model: {MODEL_ID}")

# base_model = LlamaForCausalLM.from_pretrained( # Renamed to base_model

# MODEL_ID,

# torch_dtype=dtype,

# device_map="auto",

# low_cpu_mem_usage=True,

# attn_implementation="eager" # Keep eager for compatibility during trace

# ).to(device).eval()

# config = base_model.config # Get config from base_model

# logger.info("Base model loaded.")

  

# # Instantiate the stateless wrapper

# model = StatelessLlamaWrapper(base_model).eval()

  

# # --- Calculate expected KV cache shape PER LAYER --- Needed for defining I/O

# head_dim = config.hidden_size // config.num_attention_heads

# num_kv_heads = config.num_key_value_heads

# num_layers = config.num_hidden_layers

# # Shape for ONE layer's key/value cache: (batch, num_kv_heads, seq_len, head_dim)

# # We use CONTEXT_LENGTH for the seq_len dimension in the definition

# kv_shape_per_layer = (

# BATCH_SIZE,

# num_kv_heads,

# CONTEXT_LENGTH,

# head_dim

# )

# logger.info(f"KV cache shape per layer: {kv_shape_per_layer}")

  

# # Example inputs for tracing the BASE model

# example_seq_len = 2 # Keep small

# example_input_ids = torch.zeros((BATCH_SIZE, example_seq_len), dtype=torch.int32, device=device)

# # Base model forward might only need 2D mask, but let's try with 4D causal mask first

# # If tracing fails, try a 2D mask: torch.ones(...) instead

# example_causal_mask = torch.ones(

# (BATCH_SIZE, 1, example_seq_len, example_seq_len), # Q=K=example_seq_len for base trace

# dtype=dtype,

# device=device

# )

# # Example past_key_values for tracing (Tuple of tuples (key, value) per layer)

# # Create dummy past KV tensors for the trace call

# # Shape: B, num_kv_heads, PAST_seq_len=0, head_dim

# example_past_kv_seq_len = 0 # Match lower bound of past_kv_seq_len_dim

# example_past_kv_shape = (

# BATCH_SIZE, num_kv_heads, example_past_kv_seq_len, head_dim

# )

# dummy_past_kv_tensors = []

# for _ in range(num_layers):

# dummy_key = torch.zeros(example_past_kv_shape, dtype=dtype, device=device)

# dummy_value = torch.zeros(example_past_kv_shape, dtype=dtype, device=device)

# dummy_past_kv_tensors.extend([dummy_key, dummy_value])

# dummy_past_kv_tensors = tuple(dummy_past_kv_tensors)

# logger.info(f"Created {len(dummy_past_kv_tensors)} dummy past KV tensors for tracing with shape {example_past_kv_shape}")

  

# # Define the full list of example inputs for the trace

# example_trace_inputs = (example_input_ids, example_causal_mask) + dummy_past_kv_tensors

  

# logger.info(f"Tracing the StatelessLlamaWrapper... Total inputs: {len(example_trace_inputs)}")

  

# # Trace the WRAPPER model.

# with torch.no_grad():

# # Pass the full tuple including dummy past KVs

# traced_model = torch.jit.trace(

# model,

# example_trace_inputs, # Pass the combined inputs

# check_trace=False

# )

# logger.info("Tracing complete.")

  

# # Core ML conversion setup - STATELESS

# query_length = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)

# # Full context dim for mask K/V length

# key_value_length_dim = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=1)

  

# # --- Define Inputs ---

# inputs = [

# ct.TensorType(name="inputIds", shape=(BATCH_SIZE, query_length), dtype=np.int32),

# # The mask's key/value dimension should match the stateful cache's capacity

# ct.TensorType(name="causalMask", shape=(BATCH_SIZE, 1, query_length, key_value_length_dim), dtype=np.float16),

# ]

# # Add explicit PAST KV inputs for stateless model

# # The sequence length dimension here should represent the PAST length

# past_kv_seq_len_dim = ct.RangeDim(lower_bound=0, upper_bound=CONTEXT_LENGTH-1, default=0)

# for i in range(num_layers):

# inputs.append(ct.TensorType(name=f"past_key_{i}",

# # Shape: B, num_kv_heads, PAST_seq_len, head_dim

# shape=(BATCH_SIZE, num_kv_heads, past_kv_seq_len_dim, head_dim),

# dtype=np.float16))

# inputs.append(ct.TensorType(name=f"past_value_{i}",

# # Shape: B, num_kv_heads, PAST_seq_len, head_dim

# shape=(BATCH_SIZE, num_kv_heads, past_kv_seq_len_dim, head_dim),

# dtype=np.float16))

# logger.info(f"Defined {len(inputs)} total inputs for Core ML conversion.")

  

# # --- Define Outputs ---

# # Logits output

# outputs = [ ct.TensorType(name="logits", dtype=np.float16) ]

# # Add explicit PRESENT KV outputs

# # The sequence length dimension here should represent the PRESENT (updated) length

# # Output shape should reflect the concatenation of past + current

# # Let's assume output KV cache shape matches the kv_shape_per_layer definition for simplicity

# # We might need to adjust this based on how predict handles varying shapes

# present_kv_seq_len_dim = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT_LENGTH, default=example_seq_len)

# for i in range(num_layers):

# outputs.append(ct.TensorType(name=f"present_key_{i}",

# # Shape is inferred by CoreML, do not specify

# dtype=np.float16))

# outputs.append(ct.TensorType(name=f"present_value_{i}",

# # Shape is inferred by CoreML, do not specify

# dtype=np.float16))

# logger.info(f"Defined {len(outputs)} total outputs for Core ML conversion.")

  

# logger.info("Converting to Core ML (stateless, skipping model load)...") # Updated log

# mlmodel = ct.convert(

# traced_model,

# inputs=inputs,

# outputs=outputs,

# skip_model_load=True,

# minimum_deployment_target=ct.target.iOS18,

# compute_precision=ct.precision.FLOAT16,

# compute_units=ct.ComputeUnit.ALL

# )

# logger.info("Conversion call complete.")

  

# # Add metadata

# mlmodel.author = "Meta & Core ML Tools" # Updated author

# mlmodel.license = "Llama 3.2 Community License" # Updated license

# mlmodel.version = "1.0"

# mlmodel.short_description = ( # Updated description for stateless

# f"Stateless Core ML conversion of {MODEL_ID} with FP16 precision, {CONTEXT_LENGTH}-token context, "

# f"4-bit Palettization, and 8-bit LUT quantization."

# )

# # Use consistent metadata setting

# mlmodel.user_defined_metadata["co.huggingface.exporters.name"] = MODEL_ID

# logger.info("Metadata added to model.")

  

# # Define the output path

# # output_path = "Llama3Stateful_FP16.mlpackage" # Path for previous test

# output_path = "Llama3Stateless_Pal4_LUT8.mlpackage" # New path for stateless compressed model

# logger.info(f"Target output path: {output_path}")

  

# # --- Re-enable Compression --- #

# logger.info("Starting compression: 4-bit Palettization...")

  

# # --- 4-bit Palettization ---

# palettize_op_config = ct.optimize.coreml.OpPalettizerConfig(

# nbits=4,

# mode="kmeans",

# granularity="per_grouped_channel",

# group_size=16,

# )

# palettize_config = ct.optimize.coreml.OptimizationConfig(global_config=palettize_op_config)

  

# palettized_mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=palettize_config)

# logger.info("4-bit Palettization complete.")

  

# logger.info("Starting compression: 8-bit LUT Linear Quantization...")

# # --- 8-bit LUT Linear Quantization (Joint Compression) ---

# lut_quant_op_config = ct.optimize.coreml.OpLinearQuantizerConfig(

# mode="linear_symmetric",

# dtype="int8",

# granularity="per_tensor",

# )

# lut_quant_config = ct.optimize.coreml.OptimizationConfig(global_config=lut_quant_op_config)

  

# joint_compressed_mlmodel = ct.optimize.coreml.linear_quantize_weights(

# palettized_mlmodel,

# config=lut_quant_config,

# joint_compression=True

# )

# logger.info("8-bit LUT Linear Quantization complete.")

  

# # --- Save the final compressed model ---

# logger.info(f"Saving final compressed stateless Core ML model to {output_path}...")

# joint_compressed_mlmodel.save(output_path)

# logger.info(f"Stateless Palettization and LUT Quantization complete. Model saved to {output_path}.")

  
  

# if __name__ == "__main__":

# try:

# main()

# logger.info("Script finished without errors.")

# except Exception:

# logger.error("Unhandled exception during conversion:", exc_info=True)

# sys.exit(1)

# # NO STATES 4
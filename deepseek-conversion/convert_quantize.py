import torch
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Import the PyTorch quantization tools from coremltools
# from coremltools.optimize.torch.quantization import PostTrainingQuantizer, \
#     PostTrainingQuantizerConfig

# --- Custom Op Registration for bitwise_or_ ---
# Register a conversion function for the PyTorch 'bitwise_or_' (in-place)
# and 'bitwise_or' (out-of-place) operations.
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil import Builder as mb

# Attempt to import _get_inputs, path might vary slightly across versions
try:
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
except ImportError:
    # Try an alternative common location
    try:
        from coremltools.converters.mil.frontend.torch.torch_op_registry import _get_inputs
    except ImportError:
        print("Error: Could not find _get_inputs function in coremltools.")
        print("Please check coremltools installation and version.")
        exit()

@register_torch_op(torch_alias=["bitwise_or"])
def custom_bitwise_or(context, node):
    """Maps torch.bitwise_or_ or torch.bitwise_or to mb.logical_or"""
    print(f"INFO: Using custom converter for op {node.kind} (name: {node.name})")
    # torch.bitwise_or_ expects 2 inputs: input, other
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    y = inputs[1]

    # Use the MIL builder's logical_or operation as a potential equivalent for bitwise_or on masks
    # Note: This assumes the inputs are compatible (e.g., boolean or integer masks)
    res = mb.logical_or(x=x, y=y, name=node.name)

    # Add the resulting MIL op to the graph
    context.add(res)

# --- End Custom Op Registration ---

# --- Configuration ---
# Model ID from Hugging Face
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Directory where the original model files are located
model_dir = "DeepSeek-R1-Distill-Qwen-1.5B"  # Assuming it's in a subdir
# Output Core ML model package name
# output_mlpackage_name = "NonQuantized_DeepSeekR1Qwen15B.mlpackage" # Old name
final_output_mlpackage_name = "Optimized_Symmetric_DeepSeekR1Qwen15B_8bit.mlpackage" # New name for this approach
# Sequence length for tracing and defining Core ML input shape
# We'll allow variable length up to this max in the Core ML model
trace_seq_len = 128
max_seq_len = 2048  # Max context the Core ML model will accept

print(f"Starting conversion for model: {model_id}")
print(f"Model directory: {os.path.abspath(model_dir)}")
# print(f"Output CoreML package: {output_mlpackage_name}")
print(f"Final Output Quantized CoreML package: {final_output_mlpackage_name}")

# --- 1. Load Model and Tokenizer (PyTorch/Transformers) ---
print("\n--- Loading PyTorch model and tokenizer ---")
# Ensure the model exists locally in the specified directory
if not os.path.isdir(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' not found. Please download the model first.")

# Load the model configured for causal language modeling
# trust_remote_code=True might be needed for some custom model architectures
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
# Load the tokenizer associated with the model
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Set the model to evaluation mode
# This disables dropout and other training-specific layers
model.eval()
print("Model and tokenizer loaded successfully.")

# --- 2. Define PyTorch Quantization Configuration & Apply ---
# print("\n--- Defining and Applying PyTorch Quantization (8-bit Post-Training) ---")
# # Configure Post-Training Quantization using PostTrainingQuantizerConfig
# # We target int8 weights, per-tensor granularity initially.
# ptq_config = PostTrainingQuantizerConfig.from_dict({
#     "global_config": {
#         "weight_dtype": "int8",
#         "granularity": "per_channel",
#         "quantization_scheme": "symmetric" # Options: per_tensor, per_channel, per_block
#         # "quantization_scheme": "symmetric" # Default is symmetric
#     },
#     # We need to specify which module types to target for quantization
#     # For most transformers, this is primarily torch.nn.Linear
#     # "module_type_configs": {
#     #     torch.nn.Linear: None,
#     #     torch.nn.Embedding: None
#     # "module_type_configs": {
#     #     torch.nn.Linear: {
#     #         "weight_dtype": "int8",
#     #         "granularity": "per_channel",
#     #         "quantization_scheme": "symmetric"
#     #     },
#     #     torch.nn.Embedding: {
#     #         "weight_dtype": "int8",
#     #         "granularity": "per_channel",
#     #         "quantization_scheme": "symmetric"
#     #     }
#     # }
# })
#
# # Initialize the quantizer with the model and config
# quantizer = PostTrainingQuantizer(model, ptq_config)
#
# # Apply the quantization to the PyTorch model in memory
# # This modifies the 'model' object, embedding quantization info
# quantized_pytorch_model = quantizer.compress()
# print("PyTorch model quantized successfully (in memory).")

# --- 3. Create a wrapper class for tracing ---
print("\n--- Creating a wrapper class for tracing ---")
# This wrapper will only return the logits tensor during tracing
# to avoid the DynamicCache object that causes tracing to fail
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask):
        # Call the model's forward method
        outputs = self.model(input_ids, attention_mask)
        # Only return the logits tensor, which is what we need for inference
        return outputs.logits

# Create the wrapper
# wrapped_model = ModelWrapper(quantized_pytorch_model) # Use original model now
wrapped_model = ModelWrapper(model) # Use original model now
print("Model wrapper created successfully.")

# --- 4. Prepare Example Inputs & Trace the Wrapped Model ---
print("\n--- Preparing example inputs and tracing the wrapped model ---")
# Create dummy input_ids and attention_mask for tracing
# Batch size is 1, sequence length is trace_seq_len
example_input_ids = torch.randint(0, model.config.vocab_size, (1, trace_seq_len), dtype=torch.long)
example_attention_mask = torch.ones(1, trace_seq_len, dtype=torch.long)

# Trace the wrapped model using torch.jit.trace
try:
    traced_model = torch.jit.trace(wrapped_model, (example_input_ids, example_attention_mask), strict=False)
    print("Wrapped model traced successfully.")
except Exception as e:
    print(f"Error during tracing wrapped model: {e}")
    print("Tracing failed.")
    exit() # Exit if tracing fails

# --- 5. Define Core ML Input Signatures ---
print("\n--- Defining Core ML input signatures ---")
# Define the input shapes and types for the Core ML model
# We use RangeDim to allow variable sequence lengths up to max_seq_len
coreml_input_ids = ct.TensorType(
    name="input_ids",
    shape=(1, ct.RangeDim(1, max_seq_len)), # Batch size 1, sequence length 1 to max_seq_len
    dtype=int # input_ids are usually integers
)
coreml_attention_mask = ct.TensorType(
    name="attention_mask",
    shape=(1, ct.RangeDim(1, max_seq_len)), # Batch size 1, sequence length 1 to max_seq_len
    dtype=int # attention_mask is usually integers
)

# Combine into a list
coreml_inputs = [coreml_input_ids, coreml_attention_mask]
print(f"Defined Core ML inputs: input_ids, attention_mask (dynamic shape up to {max_seq_len} tokens)")

# --- 6. Convert to Core ML (Float Version First) ---
print("\n--- Converting original model to Core ML (Float) ---")
# Convert the original traced PyTorch model to a float Core ML model
try:
    coreml_model_fp = ct.convert(
        traced_model,
        inputs=coreml_inputs,
        convert_to="mlprogram", # Use ML Program format
        minimum_deployment_target=ct.target.iOS17, # Target modern OS for better LLM support
        compute_units=ct.ComputeUnit.ALL, # Allow ANE/GPU/CPU use
    )
    print("Initial Float model conversion successful!")
except Exception as e:
    print(f"Error during initial float conversion: {e}")
    print("Conversion failed.")
    exit() # Exit if conversion fails

# --- 7. Apply Core ML Quantization ---
print("\n--- Applying 8-bit Symmetric Quantization using coremltools.optimize ---")
try:
    # Define the quantization configuration using ct.optimize.coreml
    quantization_config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            # Use the enum for data type for clarity and future-proofing
            weight_dtype=ct.optimize.coreml.ClassifierConfig.DataType.INT8
        )
        # Add op-specific configurations here if necessary later, e.g.:
        # op_type_configs={
        #     "linear": ct.optimize.coreml.OpLinearQuantizerConfig(...),
        #     "embedding": ct.optimize.coreml.OpEmbeddingQuantizerConfig(...)
        # }
    )

    # Apply quantization to the float Core ML model
    # This returns a NEW model object
    quantized_coreml_model = ct.optimize.coreml.optimize(
        coreml_model_fp,
        config=quantization_config
    )
    print("Core ML 8-bit symmetric quantization applied successfully.")
except Exception as e:
    print(f"Error during Core ML quantization: {e}")
    print("Quantization failed.")
    exit()

# --- 8. Save the Quantized Core ML Model ---
print(f"\n--- Saving Final Quantized Core ML model to {final_output_mlpackage_name} ---")
# Save the model that resulted from the ct.optimize.coreml.optimize call
quantized_coreml_model.save(final_output_mlpackage_name)
print(f"Model saved successfully to {os.path.abspath(final_output_mlpackage_name)}")

print("\n--- Conversion and Quantization Completed ---")
print("Next Steps:")
print(f"1. Add the new file '{final_output_mlpackage_name}' to your Xcode project.")
print("2. Update InferenceController.swift to use the new model class name (likely 'Optimized_Symmetric_DeepSeekR1Qwen15B_8bit').")
print("3. Try loading this new quantized model in your iOS app.")

# Remove the old print statements from the previous version
# print("1. Add this .mlpackage file to your Xcode project.")
# print("2. Try loading this non-quantized model in your iOS app.")
# print("3. If it loads (even if slow/crashing later), the issue is likely related to quantization.")
# print("4. If it *still* fails with similar errors, the issue might be model size/complexity or a fundamental conversion problem.")
# print("3. Implement or find a Swift tokenizer compatible with the model.")

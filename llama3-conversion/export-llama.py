import torch
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import coremltools as ct # Import coremltools
import coremltools.optimize as cto # Import coremltools optimization

# Import necessary components from exporters
from exporters.coreml import export
from exporters.coreml.models import LlamaCoreMLConfig
from exporters.coreml.config import InputDescription
from exporters.utils import logging

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.2-3B"
# Step 1: Output path for the initial float model
OUTPUT_PATH_FP32 = "llama3.2-3b_seq512_fp32.mlpackage"
# Step 2: Final output path for the 8-bit quantized model
OUTPUT_PATH_INT8 = "llama3.2-3b_seq512_int8.mlpackage"

FEATURE = "text-generation" # Task name
SEQUENCE_LENGTH = 512
# Step 1: Use 'float32' for the initial export via exporters
INITIAL_QUANTIZATION = "float16"
# Optional: Specify compute units (all, cpu_and_gpu, cpu_only, cpu_and_ne)
# from coremltools.converters.mil import ComputeUnit
# COMPUTE_UNITS = ComputeUnit.ALL # Default

# --- Optional: Login to Hugging Face Hub (if model is private or gated) ---
# You might need to log in if the model requires authentication
# login() # Uncomment and run if needed, enter your token

# --- Configure Logging ---
logger = logging.get_logger("exporters.coreml")
logger.setLevel(logging.INFO)
ct_logger = logging.get_logger("coremltools") # Also log coremltools
ct_logger.setLevel(logging.INFO)


print(f"Loading model and tokenizer: {MODEL_ID}...")
# Load the model and tokenizer
# Ensure torchscript=True for better tracing compatibility
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Load in fp32 initially for stability, quantization happens later
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torchscript=True, torch_dtype=torch.float32)
model.eval() # Set model to evaluation mode

print("Model and tokenizer loaded.")

# --- Custom CoreML Configuration ---
print(f"Creating custom CoreML config with sequence length {SEQUENCE_LENGTH}...")

class CustomLlamaCoreMLConfig(LlamaCoreMLConfig):
    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        # Get the default inputs
        input_descs = super().inputs
        # Modify the sequence length for input_ids
        if "input_ids" in input_descs:
            input_descs["input_ids"].sequence_length = SEQUENCE_LENGTH
            print(f"Set input_ids sequence length to {SEQUENCE_LENGTH}")
        # Modify the sequence length for attention_mask if it exists
        if "attention_mask" in input_descs:
            input_descs["attention_mask"].sequence_length = SEQUENCE_LENGTH
            print(f"Set attention_mask sequence length to {SEQUENCE_LENGTH}")
        return input_descs

    # Optional: Override outputs if needed (e.g., change output names)
    # @property
    # def outputs(self) -> OrderedDict[str, OutputDescription]:
    #     # Customize outputs here if necessary
    #     return super().outputs

# Instantiate the custom config
# Note: We pass model.config and the desired task (using the FEATURE variable)
coreml_config = CustomLlamaCoreMLConfig(model.config, task=FEATURE)

print("Custom CoreML config created.")

# --- Step 1: Export the Model using Exporters (to FP32) ---
print(f"Starting CoreML export to {OUTPUT_PATH_FP32} with {INITIAL_QUANTIZATION}...")
# It's normal to see many "TracerWarning" messages during export.
mlmodel_fp32 = export(
    preprocessor=tokenizer,
    model=model,
    config=coreml_config,
    quantize=INITIAL_QUANTIZATION, # Export as float32 initially
    # compute_units=COMPUTE_UNITS, # Uncomment to specify compute units
)
print("CoreML FP32 export function finished.")

# --- Optional: Add Metadata to FP32 model ---
print("Adding metadata to the FP32 model...")
mlmodel_fp32.short_description = f"{MODEL_ID} ({FEATURE}) - Core ML FP32, SeqLen {SEQUENCE_LENGTH}"
mlmodel_fp32.author = "Exported using Hugging Face Exporters"
# mlmodel_fp32.license = "Check model card for license" # Add appropriate license
mlmodel_fp32.version = "1.0"

# --- Save the Intermediate FP32 Model ---
print(f"Saving the intermediate FP32 Core ML model to: {OUTPUT_PATH_FP32}")
mlmodel_fp32.save(OUTPUT_PATH_FP32)
print("Intermediate FP32 model saved successfully.")

# --- Validation (Optional, on FP32 model, Requires macOS) ---
try:
    from exporters.coreml import validate_model_outputs
    print("Validating FP32 model outputs (requires macOS)...")
    validate_model_outputs(
        config=coreml_config,
        preprocessor=tokenizer,
        original_model=model, # Compare against original PyTorch model
        coreml_model=mlmodel_fp32,
        atol=coreml_config.atol_for_validation, # Use default tolerance from config
    )
    print("FP32 Model validation successful.")
except ImportError:
    print("Skipping FP32 validation: coremltools not fully available or not on macOS.")
except Exception as e:
    print(f"FP32 Model validation failed: {e}")


# --- Step 2: Quantize the FP32 Core ML Model to INT8 using coremltools ---
print(f"\nLoading FP32 model for 8-bit weight quantization: {OUTPUT_PATH_FP32}")
# Load the previously saved FP32 model
model_to_quantize = ct.models.MLModel(OUTPUT_PATH_FP32)

print("Configuring 8-bit quantization...")
# Configure linear 8-bit symmetric quantization for weights
# 'linear_symmetric' uses per-channel scale, no zero-point (often good for performance)
# 'mode="linear"' uses scale and zero-point (might give slightly better accuracy)
op_config = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", weight_dtype="int8")
# Apply this config globally to all applicable layers
config = cto.coreml.OptimizationConfig(global_config=op_config)

print("Applying 8-bit weight quantization...")
# Apply the quantization
# This returns a *new* model object with quantized weights
model_int8 = cto.coreml.linear_quantize_weights(model_to_quantize, config=config)
print("8-bit weight quantization finished.")

# --- Optional: Add/Update Metadata for INT8 model ---
print("Adding metadata to the INT8 model...")
# You can copy metadata from the original or set new ones
model_int8.short_description = f"{MODEL_ID} ({FEATURE}) - Core ML INT8 Weights, SeqLen {SEQUENCE_LENGTH}"
model_int8.author = "Quantized using coremltools"
# model_int8.license = model_to_quantize.license # Copy license if needed
model_int8.version = "1.0-int8"

# --- Save the Final INT8 Model ---
print(f"Saving the final INT8 Core ML model to: {OUTPUT_PATH_INT8}")
model_int8.save(OUTPUT_PATH_INT8)
print("Final INT8 model saved successfully.")

# Note: Validation after int8 quantization is more complex as the outputs
# will differ more significantly from the original fp32 PyTorch model.
# You would typically validate performance/accuracy on a downstream task.

print("\nScript finished.")

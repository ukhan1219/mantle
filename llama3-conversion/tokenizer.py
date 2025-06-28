from transformers import AutoTokenizer
import os

hf_repo_id = "meta-llama/Llama-3.2-1B"
output_dir = "./llama1_tokenizer_bundle" # Choose a directory name

print(f"Downloading tokenizer for {hf_repo_id} to {output_dir}...")
try:
    # This requires you to be logged in via huggingface-cli
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer files saved successfully to {output_dir}")
    print("Files:")
    for filename in os.listdir(output_dir):
        print(f"- {filename}")
except Exception as e:
    print(f"Error downloading tokenizer: {e}")
    print("Please ensure you are logged in via 'huggingface-cli login' and have access to the repository.")

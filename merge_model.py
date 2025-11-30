"""
Merge LoRA weights with base model and push to Hugging Face Hub
Run this on your VM after training completes
"""

import os
import glob
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

# Configuration
LORA_OUTPUT_DIR = "./whisper-svarah-lora"
MERGED_OUTPUT_DIR = "./whisper-hindi-merged"

print("Loading base model...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA weights from best checkpoint...")
# Find the best checkpoint (looks for checkpoint-* directories)
checkpoint_dirs = sorted(glob.glob(os.path.join(LORA_OUTPUT_DIR, "checkpoint-*")))
if not checkpoint_dirs:
    raise FileNotFoundError(
        f"No checkpoint directories found in {LORA_OUTPUT_DIR}. "
        "Make sure training has completed and checkpoints were saved."
    )

# Use the latest checkpoint (highest step number)
checkpoint_path = checkpoint_dirs[-1]
print(f"Using checkpoint: {checkpoint_path}")
model = PeftModel.from_pretrained(base_model, checkpoint_path)

print("Merging LoRA weights with base model...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained(MERGED_OUTPUT_DIR)

print("Saving processor...")
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3-turbo",
    language="hindi",
    task="transcribe"
)
processor.save_pretrained(MERGED_OUTPUT_DIR)

print(f"\n✅ Model merged and saved to {MERGED_OUTPUT_DIR}")
print("\nTo push to Hugging Face Hub, run:")
print("  huggingface-cli login")
print(f"  huggingface-cli upload chaudharyritik/whisper-hindi-v1 {MERGED_OUTPUT_DIR}")

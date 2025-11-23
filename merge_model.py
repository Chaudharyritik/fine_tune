"""
Merge LoRA weights with base model and push to Hugging Face Hub
Run this on your VM after training completes
"""

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

print("Loading base model...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA weights from best checkpoint...")
# Load the best checkpoint (checkpoint-500 based on your training)
model = PeftModel.from_pretrained(base_model, "./whisper-svarah-lora/checkpoint-500")

print("Merging LoRA weights with base model...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained("./whisper-hindi-merged")

print("Saving processor...")
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3-turbo",
    language="hindi",
    task="transcribe"
)
processor.save_pretrained("./whisper-hindi-merged")

print("\n✅ Model merged and saved to ./whisper-hindi-merged")
print("\nTo push to Hugging Face Hub, run:")
print("  huggingface-cli login")
print("  huggingface-cli upload chaudharyritik/whisper-hindi-v1 ./whisper-hindi-merged")

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os

# Configuration
BASE_MODEL_ID = "chaudharyritik1/whisper-hindi-v1"
LORA_OUTPUT_DIR = os.getenv("LORA_OUTPUT_DIR", "./whisper-hindi-cv-lora")
MERGED_OUTPUT_DIR = "./whisper-hindi-cv-merged"

def merge_model():
    print(f"Loading base model: {BASE_MODEL_ID}")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA weights from: {LORA_OUTPUT_DIR}")
    # Load the best checkpoint or the final model
    # Assuming the trainer saved the final model in the output dir
    # If checkpoints exist, we might want to pick the best one, but let's default to the output dir first
    try:
        model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    except Exception as e:
        print(f"Could not load from {LORA_OUTPUT_DIR}, trying to find latest checkpoint...")
        # Find latest checkpoint
        checkpoints = [d for d in os.listdir(LORA_OUTPUT_DIR) if d.startswith("checkpoint-")]
        if not checkpoints:
            raise ValueError("No checkpoints found!")
        
        # Sort by step number
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        checkpoint_path = os.path.join(LORA_OUTPUT_DIR, latest_checkpoint)
        print(f"Loading from latest checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)

    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_OUTPUT_DIR}")
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)

    print("Saving processor...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language="hindi", task="transcribe")
    processor.save_pretrained(MERGED_OUTPUT_DIR)

    print("\n✅ Model merged successfully!")
    print(f"You can now push '{MERGED_OUTPUT_DIR}' to Hugging Face Hub.")

if __name__ == "__main__":
    merge_model()

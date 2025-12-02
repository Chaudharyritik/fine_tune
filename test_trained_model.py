import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from peft import PeftModel
import librosa
import os
import glob

# Configuration
LORA_OUTPUT_DIR = "./whisper-hindi-cv-lora"
BASE_MODEL_ID = "openai/whisper-large-v3-turbo"
LANGUAGE = "hindi"
TASK = "transcribe"
AUDIO_FILE = "test_hindi.mp3"  # Make sure this file exists

def get_best_checkpoint(checkpoint_dir):
    # Find the latest checkpoint
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Use the latest checkpoint
    best_checkpoint = checkpoints[-1]
    print(f"Using checkpoint: {best_checkpoint}")
    return best_checkpoint

def main():
    # 1. Load Model
    print("Loading model...")
    try:
        checkpoint_path = get_best_checkpoint(LORA_OUTPUT_DIR)
    except FileNotFoundError:
        print(f"❌ Error: No checkpoints found in {LORA_OUTPUT_DIR}. Did you train the model?")
        return

    # Load base model
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language=LANGUAGE, task=TASK)
    
    # 2. Load Audio
    print(f"Loading audio: {AUDIO_FILE}...")
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Error: Audio file {AUDIO_FILE} not found.")
        return

    # Load and resample to 16kHz
    audio_array, sampling_rate = librosa.load(AUDIO_FILE, sr=16000)
    
    # 3. Transcribe
    print("Transcribing...")
    input_features = processor(
        audio_array, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_features.to(model.device).to(torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(input_features, language=LANGUAGE)
        
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*50)
    print("📝 Transcription Result")
    print("="*50)
    print(transcription)
    print("="*50)

if __name__ == "__main__":
    main()

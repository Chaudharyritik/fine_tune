import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig
from transformers import BitsAndBytesConfig
import evaluate
import jiwer
from tqdm import tqdm
import os
import glob
import numpy as np

# Configuration
LORA_OUTPUT_DIR = "./whisper-hindi-cv-lora"
BASE_MODEL_ID = "openai/whisper-large-v3-turbo"
LANGUAGE = "hindi"
TASK = "transcribe"
BATCH_SIZE = 8
MAX_EVAL_SAMPLES = 500  # Evaluate on a subset for speed, or None for full test set

def load_dataset_for_eval():
    print("Loading test dataset...")
    # Use the same dataset logic as training
    dataset_configs = [
        {"name": "fixie-ai/common_voice_17_0", "config": "hi", "split": "test", "trust_remote_code": True, "verification_mode": "no_checks"},
        {"name": "mozilla-foundation/common_voice_11_0", "config": "hi", "split": "test", "trust_remote_code": True},
        {"name": "ai4bharat/IndicVoices", "config": "hi", "split": "test", "trust_remote_code": True},
        {"name": "ai4bharat/kathbath", "config": "hindi", "split": "test", "trust_remote_code": True},
    ]

    dataset = None
    for config in dataset_configs:
        try:
            print(f"Trying to load {config['name']}...")
            load_kwargs = {k: v for k, v in config.items() if k not in ["name", "config"]}
            dataset = load_dataset(config["name"], config["config"], **load_kwargs)
            print(f"✅ Loaded {config['name']}")
            break
        except Exception as e:
            print(f"❌ Failed to load {config['name']}: {e}")
            continue
    
    if dataset is None:
        raise RuntimeError("Could not load any test dataset.")
    
    if MAX_EVAL_SAMPLES and len(dataset) > MAX_EVAL_SAMPLES:
        print(f"Subsampling test set to {MAX_EVAL_SAMPLES} samples...")
        dataset = dataset.select(range(MAX_EVAL_SAMPLES))
        
    return dataset

def get_best_checkpoint(checkpoint_dir):
    # Find the latest checkpoint
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Try to find the best model from trainer state if available, else use latest
    # For now, simply using the latest checkpoint which is often the best if load_best_model_at_end=True
    best_checkpoint = checkpoints[-1]
    print(f"Using checkpoint: {best_checkpoint}")
    return best_checkpoint

def main():
    # 1. Load Dataset
    dataset = load_dataset_for_eval()
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 2. Load Model
    print("Loading model...")
    checkpoint_path = get_best_checkpoint(LORA_OUTPUT_DIR)
    
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
    
    # 3. Evaluation Loop
    print("Starting evaluation...")
    predictions = []
    references = []
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    for i, batch in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
        audio = batch["audio"]
        
        # Get ground truth text
        text_cols = ["sentence", "transcription", "text"]
        reference_text = None
        for col in text_cols:
            if col in batch:
                reference_text = batch[col]
                break
        
        if reference_text is None:
            continue
            
        references.append(reference_text)
        
        # Transcribe
        input_features = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_tensors="pt"
        ).input_features.to(model.device)
        
        # Cast input to match model dtype (usually float16 for 8-bit/fp16 models)
        if model.dtype == torch.float16 or model.dtype == torch.bfloat16:
            input_features = input_features.to(model.dtype)
        
        with torch.no_grad():
            generated_ids = model.generate(input_features, language=LANGUAGE)
            
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(transcription)
        
        # Print first 5 samples for inspection
        if i < 5:
            print(f"\nSample {i+1}:")
            print(f"Ref:  {reference_text}")
            print(f"Pred: {transcription}")
            print("-" * 50)

    # 4. Compute Metrics
    print("\nComputing metrics...")
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)
    
    print("\n" + "="*50)
    print("📊 Evaluation Results")
    print("="*50)
    print(f"Number of samples: {len(dataset)}")
    print(f"WER (Word Error Rate):      {wer:.2f}%")
    print(f"CER (Character Error Rate): {cer:.2f}%")
    
    # Accuracy roughly (100 - WER), though WER can be > 100
    accuracy = max(0, 100 - wer)
    print(f"Approx. Accuracy:           {accuracy:.2f}%")
    print("="*50)
    
    # Save results
    with open("evaluation_results.txt", "w") as f:
        f.write(f"WER: {wer:.2f}%\n")
        f.write(f"CER: {cer:.2f}%\n")
        f.write("\nSample Predictions:\n")
        for i in range(min(20, len(predictions))):
            f.write(f"\nRef:  {references[i]}\n")
            f.write(f"Pred: {predictions[i]}\n")
            f.write("-" * 30 + "\n")
            
    print("Results saved to evaluation_results.txt")

if __name__ == "__main__":
    main()


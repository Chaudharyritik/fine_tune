"""
Fine-tune Whisper on Local Common Voice (cv-corpus-23.0) Hindi Dataset
Uses proper train/dev/test splits from TSV files
Conservative settings to preserve FLEURS-trained model learning
"""

import torch
from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import jiwer
import os
import pandas as pd
import soundfile as sf
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
LOCAL_CV_PATH = os.getenv("LOCAL_CV_PATH", "/home/fine_tune/cv-corpus-23.0-2025-09-05/hi")
FINE_TUNED_MODEL_ID = "chaudharyritik1/whisper-hindi-v1"
OUTPUT_DIR = "./whisper-hindi-cv-lora"

# Validation thresholds
MIN_AUDIO_DURATION = 0.1  # seconds
MAX_AUDIO_DURATION = 30.0  # seconds
MIN_TRANSCRIPTION_LENGTH = 1  # characters
MAX_TRANSCRIPTION_LENGTH = 500  # characters

# Training hyperparameters (conservative to preserve FLEURS learning)
LEARNING_RATE = 1e-5
MAX_STEPS = 1500
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
WARMUP_STEPS = 50

# =============================================================================
# Validation Functions
# =============================================================================
def is_valid_sample(audio_path: Optional[str] = None, audio_array: Optional[np.ndarray] = None, 
                    sampling_rate: Optional[int] = None, transcription: Optional[str] = None) -> bool:
    """Validate a dataset sample before training."""
    # Check transcription
    if transcription is None or not isinstance(transcription, str):
        return False
    
    transcription = transcription.strip()
    if len(transcription) < MIN_TRANSCRIPTION_LENGTH or len(transcription) > MAX_TRANSCRIPTION_LENGTH:
        return False
    
    # Check audio array if provided
    if audio_array is not None and sampling_rate is not None:
        duration = len(audio_array) / sampling_rate
        if duration < MIN_AUDIO_DURATION or duration > MAX_AUDIO_DURATION:
            return False
        if np.all(audio_array == 0) or not np.isfinite(audio_array).all():
            return False
    
    # Check audio path if provided
    if audio_path is not None and not os.path.exists(audio_path):
        return False
    
    return True

# =============================================================================
# Load Local Common Voice Dataset
# =============================================================================
def load_local_cv_split(cv_path: str, split: str = "train") -> Optional[Dataset]:
    """
    Load a specific split from local Common Voice dataset.
    
    Args:
        cv_path: Path to the Common Voice language directory (e.g., /path/to/hi/)
        split: One of "train", "dev", or "test"
    
    Returns:
        HuggingFace Dataset or None if loading fails
    """
    split_to_file = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "validation": "dev.tsv",
        "test": "test.tsv"
    }
    
    if split not in split_to_file:
        print(f"Unknown split: {split}. Using train.tsv")
        split = "train"
    
    tsv_file = split_to_file[split]
    tsv_path = os.path.join(cv_path, tsv_file)
    
    if not os.path.exists(tsv_path):
        print(f"TSV file not found: {tsv_path}")
        return None
    
    clips_dir = os.path.join(cv_path, "clips")
    if not os.path.exists(clips_dir):
        print(f"Clips directory not found: {clips_dir}")
        return None
    
    print(f"Loading {split} split from: {tsv_file}")
    
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        
        if "path" not in df.columns or "sentence" not in df.columns:
            print(f"TSV missing required columns. Found: {df.columns.tolist()}")
            return None
        
        data = []
        valid_count = 0
        invalid_count = 0
        
        for idx, row in df.iterrows():
            audio_filename = row["path"]
            transcription = str(row["sentence"]).strip()
            audio_path = os.path.join(clips_dir, audio_filename)
            
            # Quick validation
            if not os.path.exists(audio_path):
                invalid_count += 1
                continue
            
            if not is_valid_sample(transcription=transcription):
                invalid_count += 1
                continue
            
            # Load and validate audio
            try:
                audio_data, sr = sf.read(audio_path)
                
                if not is_valid_sample(audio_array=audio_data, sampling_rate=sr, transcription=transcription):
                    invalid_count += 1
                    continue
                
                data.append({
                    "audio": {"path": audio_path, "array": audio_data, "sampling_rate": sr},
                    "sentence": transcription
                })
                valid_count += 1
                
            except Exception as e:
                invalid_count += 1
                continue
        
        if len(data) == 0:
            print(f"No valid samples found in {split} split")
            return None
        
        print(f"  ✅ Loaded {valid_count} valid samples, skipped {invalid_count} invalid")
        
        dataset = Dataset.from_list(data)
        return dataset
        
    except Exception as e:
        print(f"Error loading {split} split: {e}")
        return None

# =============================================================================
# Main Script
# =============================================================================
from huggingface_hub import whoami
try:
    user_info = whoami()
    print(f"Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Not logged in or error checking auth: {e}")

print("="*60)
print("Fine-tuning Whisper on Local Common Voice Hindi Dataset")
print("="*60)
print(f"Dataset path: {LOCAL_CV_PATH}")
print(f"Base model: {FINE_TUNED_MODEL_ID}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Max steps: {MAX_STEPS}")
print("="*60)

# Load train and dev splits
print("\n1. Loading dataset splits...")
train_dataset = load_local_cv_split(LOCAL_CV_PATH, "train")
dev_dataset = load_local_cv_split(LOCAL_CV_PATH, "dev")

if train_dataset is None:
    raise RuntimeError(f"Failed to load training dataset from {LOCAL_CV_PATH}")

if dev_dataset is None:
    print("⚠️  Dev dataset not available, will create validation split from training")
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    dev_dataset = split["test"]

print(f"\n✅ Dataset loaded:")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Validation: {len(dev_dataset)} samples")

# Cast audio columns
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
dev_dataset = dev_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load processor
print("\n2. Loading processor...")
processor = WhisperProcessor.from_pretrained(FINE_TUNED_MODEL_ID, language="hindi", task="transcribe")

# Preprocessing function
def prepare_dataset(batch):
    """Preprocess a single sample."""
    try:
        audio = batch["audio"]
        if audio is None:
            return None
        
        audio_array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
        
        if audio_array is None or sampling_rate is None:
            return None
        
        # Compute features
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=sampling_rate
        ).input_features[0]
        
        # Get transcription
        text = batch.get("sentence", "").strip()
        if len(text) == 0:
            return None
        
        # Tokenize
        batch["labels"] = processor.tokenizer(text).input_ids
        if len(batch["labels"]) == 0:
            return None
        
        return batch
    except Exception:
        return None

# Apply preprocessing
print("\n3. Preprocessing datasets...")

def preprocess_and_filter(dataset, desc="Processing"):
    """Preprocess dataset and filter invalid samples."""
    original_size = len(dataset)
    
    # Apply preprocessing
    processed = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        num_proc=1,
        desc=desc
    )
    
    # Filter None results
    processed = processed.filter(
        lambda x: x is not None and 
                  x.get("input_features") is not None and 
                  x.get("labels") is not None and
                  len(x.get("labels", [])) > 0
    )
    
    filtered_size = len(processed)
    removed = original_size - filtered_size
    print(f"  {desc}: {original_size} -> {filtered_size} samples (removed {removed})")
    
    return processed

train_dataset = preprocess_and_filter(train_dataset, "Train preprocessing")
dev_dataset = preprocess_and_filter(dev_dataset, "Dev preprocessing")

print(f"\n✅ Final dataset sizes:")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Validation: {len(dev_dataset)} samples")

# Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=processor.tokenizer.bos_token_id
)

# Load model
print("\n4. Loading model with 8-bit quantization...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = WhisperForConditionalGeneration.from_pretrained(
    FINE_TUNED_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model = prepare_model_for_kbit_training(model)

# Apply LoRA
print("5. Applying LoRA...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Metrics
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    """Compute WER and CER metrics."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * jiwer.cer(label_str, pred_str)

    return {"wer": wer, "cer": cer}

# Training arguments
print("\n6. Setting up training...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    lr_scheduler_type="linear",
    gradient_checkpointing=True,
    bf16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    label_names=["labels"],
    save_total_limit=3,
)

# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# Train
print("\n" + "="*60)
print("Starting training...")
print("="*60)
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(dev_dataset)}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Max steps: {MAX_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print("="*60)

trainer.train()

print("\n" + "="*60)
print("Training completed!")
print(f"Best model saved to: {OUTPUT_DIR}")
print("="*60)

# Final evaluation
print("\nRunning final evaluation on validation set...")
eval_results = trainer.evaluate()
print(f"Final WER: {eval_results['eval_wer']:.2f}%")
print(f"Final CER: {eval_results['eval_cer']:.2f}%")


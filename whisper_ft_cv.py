import os
import datetime
import torch
import pandas as pd
from datasets import Dataset, Audio, DatasetDict
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
from typing import Any, Dict, List, Union

# Configuration
DEFAULT_CV_PATH = "/Users/jaidratsingh/Downloads/cv-corpus-23.0-2025-09-05/hi"
CV_PATH = os.getenv("CV_PATH", DEFAULT_CV_PATH)
MODEL_ID = "chaudharyritik1/whisper-hindi-v1"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"./whisper-hindi-cv-lora_{timestamp}"
print(f"Output directory set to: {OUTPUT_DIR}")

def load_local_common_voice(cv_path):
    """Load Common Voice dataset from local directory."""
    # Debug: Check if path exists
    if not os.path.exists(cv_path):
        print(f"❌ Error: CV_PATH does not exist: {cv_path}")
        return DatasetDict()

    print(f"Checking contents of {cv_path}...")
    try:
        print(f"Files: {os.listdir(cv_path)}")
    except Exception as e:
        print(f"Could not list directory: {e}")

    data = {}
    for split in ["train", "dev", "test"]:
        tsv_path = os.path.join(cv_path, f"{split}.tsv")
        if not os.path.exists(tsv_path):
            print(f"Warning: {split}.tsv not found in {cv_path}")
            continue
            
        df = pd.read_csv(tsv_path, sep="\t")
        # Add audio path
        df["audio"] = df["path"].apply(lambda x: os.path.join(cv_path, "clips", x))
        # Rename sentence to transcription (standardize)
        df = df.rename(columns={"sentence": "transcription"})
        
        # Create dataset
        ds = Dataset.from_pandas(df)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        data[split] = ds
    
    if "train" not in data:
        raise ValueError(f"❌ Failed to load 'train' split from {cv_path}. Please check the path.")
        
    return DatasetDict(data)

# 1. Load Dataset
print(f"Loading Common Voice dataset from {CV_PATH}...")
dataset = load_local_common_voice(CV_PATH)
print(f"Dataset loaded: {dataset}")

# 2. Model and Processor
print(f"Loading processor for {MODEL_ID}...")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="hindi", task="transcribe")

# 3. Preprocessing
def prepare_dataset(batch):
    # Load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    
    # Compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # Encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

# Apply preprocessing with aggressive memory management
print("Preprocessing dataset...")
column_names = dataset["train"].column_names

# Process each split separately and save to disk to clear memory
processed_datasets = {}

for split in dataset.keys():
    print(f"Processing split: {split}")
    # Select the split
    split_ds = dataset[split]
    
    # Map with memory optimizations
    processed_split = split_ds.map(
        prepare_dataset, 
        remove_columns=column_names, 
        num_proc=1,
        batched=False,
        writer_batch_size=50, # Even smaller write batch
        keep_in_memory=False,
        load_from_cache_file=True # Use cache
    )
    
    # Force save to disk and reload to free RAM
    save_path = os.path.join(OUTPUT_DIR, f"processed_{split}")
    processed_split.save_to_disk(save_path)
    
    # Clear memory
    del split_ds
    del processed_split
    import gc
    gc.collect()
    
    # Load back from disk
    from datasets import load_from_disk
    processed_datasets[split] = load_from_disk(save_path)

dataset = DatasetDict(processed_datasets)

# 4. Data Collator
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

# 5. Load Model with 8-bit Quantization and LoRA
print(f"Loading model {MODEL_ID} with 8-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA Config
print("Applying LoRA...")
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# 6. Metrics
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# 7. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # Keep at 4 to prevent OOM
    gradient_accumulation_steps=4, # Increase to 4 -> Effective batch size = 16
    learning_rate=1e-5, 
    warmup_steps=100,
    num_train_epochs=5, # Train for 5 full epochs
    # max_steps=1000, # Removed in favor of epochs
    gradient_checkpointing=True,
    bf16=True, 
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False, 
    label_names=["labels"], 
    save_total_limit=3, # Keep only last 3 checkpoints
)

# 8. Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"] if "dev" in dataset else dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("Starting training...")
trainer.train()

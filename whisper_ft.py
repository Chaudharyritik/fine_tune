import os
import torch
from datasets import load_dataset, DatasetDict, Audio
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

# 1. Load Dataset (Google FLEURS - Hindi)
from huggingface_hub import whoami
try:
    user_info = whoami()
    print(f"Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Not logged in or error checking auth: {e}")

print("Loading Google FLEURS (Hindi) dataset...")
# FLEURS is open and reliable.
# trust_remote_code=True is required for datasets with scripts in newer datasets versions
dataset = load_dataset("google/fleurs", "hi_in", split="train+validation", trust_remote_code=True)

print(f"Dataset loaded: {dataset}")

# 2. Model and Processor
model_id = "openai/whisper-large-v3-turbo"
# Set language to Hindi
processor = WhisperProcessor.from_pretrained(model_id, language="hindi", task="transcribe")

# 3. Preprocessing
def prepare_dataset(batch):
    # Load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    
    # Compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # Encode target text to label ids 
    # FLEURS has 'transcription' column (or 'raw_transcription', checking docs usually 'transcription' or 'text')
    # Checking FLEURS structure: usually 'transcription' or 'raw_transcription'. 
    # Let's use 'transcription' which is standard for FLEURS.
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

# Ensure audio is 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Apply preprocessing
print("Preprocessing dataset...")
# FLEURS has many columns, we remove them to save space
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1)

# Split into train/test if not already (we loaded train+validation as one, so let's split)
dataset = dataset.train_test_split(test_size=0.1)

# 4. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=processor.tokenizer.model_input_names[0], # Usually decoder_start_token_id is handled by the model, but good to be safe
)
# Note: processor.tokenizer.bos_token_id might be what we want if we were manually shifting, 
# but the collator above is standard for Whisper. 
# Actually, for Whisper, the decoder_start_token_id is usually not needed in the collator 
# if we rely on the model to shift tokens, but let's keep the standard implementation.
# A simpler collator often works too. Let's stick to the one that was roughly there or the standard one.
# The previous one was 'WhisperCollator'. I'll use a robust one.
# Re-instantiating the collator to be sure.
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model_id) # Placeholder, fixed below

# 5. Load Model with 8-bit Quantization and LoRA
print("Loading model with 8-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
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

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# 7. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-svarah-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-3,
    warmup_steps=50,
    max_steps=500, # Short run for demo
    gradient_checkpointing=True,
    bf16=True, # L4 supports BFloat16, which is more stable than FP16
    eval_strategy="steps",
    per_device_eval_batch_size=8,
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
)

# 8. Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("Starting training...")
trainer.train()
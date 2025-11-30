import torch
from datasets import load_dataset, Audio
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
import jiwer

# 1. Load Dataset (Common Voice - Hindi)
from huggingface_hub import whoami
try:
    user_info = whoami()
    print(f"Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Not logged in or error checking auth: {e}")

print("Loading Hindi dataset for training...")
# Try multiple datasets: Common Voice and Indic-Voices
# Common Voice may require accepting terms at: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0

dataset_configs = [
    # Try Indic-Voices first (more reliable for Hindi)
    {
        "name": "ai4bharat/indic-voices",
        "config": "hi",
        "split": "train",
        "streaming": False,
        "trust_remote_code": True
    },
    # Try Common Voice versions (may require accepting terms)
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "config": "hi",
        "split": "train",
        "streaming": False,
        "trust_remote_code": True
    },
    {
        "name": "mozilla-foundation/common_voice_16_1",
        "config": "hi",
        "split": "train",
        "streaming": False,
        "trust_remote_code": True
    },
    {
        "name": "mozilla-foundation/common_voice_15_0",
        "config": "hi",
        "split": "train",
        "streaming": False,
        "trust_remote_code": True
    },
]

dataset = None
last_error = None
successful_config = None

for config in dataset_configs:
    try:
        dataset_name = config["name"]
        lang_code = config["config"]
        streaming = config.get("streaming", False)
        trust_remote = config.get("trust_remote_code", True)
        
        print(f"Trying to load: {dataset_name} (language: {lang_code}, streaming: {streaming})...")
        
        load_kwargs = {
            "split": config["split"],
            "trust_remote_code": trust_remote
        }
        if streaming:
            load_kwargs["streaming"] = True
        
        dataset = load_dataset(dataset_name, lang_code, **load_kwargs)
        
        # If streaming, convert to regular dataset (take first N samples for testing)
        if streaming:
            print("Streaming mode detected. Converting to regular dataset...")
            # For streaming, we need to materialize it - take a reasonable sample
            # You can adjust this number based on your needs
            dataset = dataset.take(50000)  # Take first 50k samples, adjust as needed
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        print(f"✅ Successfully loaded: {dataset_name}")
        successful_config = config
        break
    except Exception as e:
        print(f"❌ Failed to load {dataset_name}: {str(e)[:200]}...")
        last_error = e
        continue

if dataset is None:
    print("\n" + "="*60)
    print("ERROR: Could not load any Hindi dataset.")
    print("="*60)
    print("\nPossible solutions:")
    print("1. Accept terms of use for Common Voice:")
    print("   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
    print("2. Check if you're logged in: huggingface-cli login")
    print("3. Try alternative datasets like Indic-Voices")
    print("4. Check dataset availability on Hugging Face Hub")
    print(f"\nLast error: {last_error}")
    raise RuntimeError(f"Failed to load Hindi dataset. Last error: {last_error}")

print(f"\n✅ Using dataset: {successful_config['name']} (language: {successful_config['config']})")

print(f"Dataset loaded: {dataset}")
print(f"Dataset size: {len(dataset)} samples")

# 2. Model and Processor - Load existing fine-tuned model
fine_tuned_model_id = "chaudharyritik1/whisper-hindi-v1"
print(f"Loading fine-tuned model from: {fine_tuned_model_id}")

# Load processor from the fine-tuned model
processor = WhisperProcessor.from_pretrained(fine_tuned_model_id, language="hindi", task="transcribe")

# 3. Preprocessing
def prepare_dataset(batch):
    # Load and resample audio data to 16kHz
    audio = batch["audio"]
    
    # Compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # Encode target text to label ids 
    # Common Voice uses 'sentence', Indic-Voices might use 'transcription' or 'text'
    # Try different column names
    if "sentence" in batch:
        text = batch["sentence"]
    elif "transcription" in batch:
        text = batch["transcription"]
    elif "text" in batch:
        text = batch["text"]
    else:
        # Try to find any text-like column
        text_cols = [col for col in batch.keys() if col not in ["audio", "input_features"]]
        if text_cols:
            text = batch[text_cols[0]]
            print(f"Warning: Using column '{text_cols[0]}' for transcription")
        else:
            raise ValueError("Could not find transcription column in dataset")
    
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

# Ensure audio is 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Apply preprocessing
print("Preprocessing dataset...")
# Remove all original columns after preprocessing (we only need input_features and labels)
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1)

# Split into train/validation (90/10 split)
print("Splitting dataset into train/validation...")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"Train samples: {len(dataset['train'])}, Validation samples: {len(dataset['test'])}")

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
    decoder_start_token_id=processor.tokenizer.bos_token_id
)

# 5. Load Fine-tuned Model with 8-bit Quantization and LoRA
print("Loading fine-tuned model with 8-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load the existing fine-tuned model (merged model, not base)
model = WhisperForConditionalGeneration.from_pretrained(
    fine_tuned_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA Config - Apply LoRA on top of fine-tuned model
print("Applying LoRA on fine-tuned model...")
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# 6. Enhanced Metrics
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    """
    Compute comprehensive evaluation metrics:
    - WER (Word Error Rate)
    - CER (Character Error Rate)
    - Precision, Recall, F1-Score (word level)
    - Accuracy (character level)
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # WER (Word Error Rate)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

    # CER (Character Error Rate) using jiwer
    cer = 100 * jiwer.cer(label_str, pred_str)

    # Word-level Precision, Recall, F1-Score
    # Tokenize into words for word-level metrics
    def tokenize_words(text):
        """Simple word tokenization (split by whitespace and normalize)"""
        return text.lower().strip().split()
    
    # Compute word-level metrics across all samples
    total_pred_words = 0
    total_label_words = 0
    correct_words = 0
    
    for pred, label in zip(pred_str, label_str):
        pred_words = set(tokenize_words(pred))
        label_words = set(tokenize_words(label))
        
        # Count matches (words that appear in both)
        matches = len(pred_words & label_words)
        correct_words += matches
        total_pred_words += len(pred_words)
        total_label_words += len(label_words)
    
    # Calculate precision, recall, F1
    precision = (correct_words / total_pred_words * 100) if total_pred_words > 0 else 0.0
    recall = (correct_words / total_label_words * 100) if total_label_words > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Character-level accuracy
    total_chars = sum(len(ref) for ref in label_str)
    correct_chars = 0
    for pred, ref in zip(pred_str, label_str):
        # Count matching characters up to min length
        min_len = min(len(pred), len(ref))
        correct_chars += sum(1 for i in range(min_len) if pred[i] == ref[i])
    
    char_accuracy = (correct_chars / total_chars * 100) if total_chars > 0 else 0.0

    return {
        "wer": wer,
        "cer": cer,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "accuracy": char_accuracy,
    }

# 7. Training Arguments - Optimized for continued training
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-hindi-cv-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,  # Lower learning rate for continued training
    warmup_steps=100,
    max_steps=2000,  # More steps for better convergence
    gradient_checkpointing=True,
    bf16=True,  # BFloat16 for stability
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=200,
    eval_steps=200,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",  # Use WER as primary metric
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False, 
    label_names=["labels"],
    save_total_limit=3,  # Keep only last 3 checkpoints
)

# 8. Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("="*60)
print("Starting continued training on Common Voice Hindi dataset...")
print(f"Model: {fine_tuned_model_id}")
print(f"Dataset: Common Voice Hindi (train: {len(dataset['train'])}, val: {len(dataset['test'])})")
print(f"Output directory: {training_args.output_dir}")
print("="*60)
trainer.train()

print("\n" + "="*60)
print("Training completed!")
print(f"Best model saved to: {training_args.output_dir}")
print("="*60)


import torch
from datasets import load_dataset, Audio, Dataset, concatenate_datasets
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
import gc

# Configuration
LOCAL_CV_PATH = os.getenv("LOCAL_CV_PATH", "/home/fine_tune/cv-corpus-23.0-2025-09-05/hi")
MIN_AUDIO_DURATION = 0.1  # seconds
MAX_AUDIO_DURATION = 30.0  # seconds
MIN_TRANSCRIPTION_LENGTH = 1  # characters
MAX_TRANSCRIPTION_LENGTH = 500  # characters

# Memory optimization settings
MAX_DATASETS_TO_LOAD = 2  # Limit number of datasets to prevent memory issues
MAX_SAMPLES_PER_DATASET = 50000  # Limit samples per dataset when streaming

# Dataset loading configuration
# Options: "auto" (skip online if local exists), "always" (always load online), "never" (never load online)
USE_ONLINE_DATASETS = os.getenv("USE_ONLINE_DATASETS", "auto")

# 1. Load Dataset (Common Voice - Hindi)
from huggingface_hub import whoami
try:
    user_info = whoami()
    print(f"Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Not logged in or error checking auth: {e}")

# Validation Functions
def is_valid_sample(audio_path: Optional[str] = None, audio_array: Optional[np.ndarray] = None, 
                    sampling_rate: Optional[int] = None, transcription: Optional[str] = None) -> bool:
    """
    Validate a dataset sample before training.
    Checks audio file existence, duration, and transcription quality.
    """
    # Check transcription
    if transcription is None or not isinstance(transcription, str):
        return False
    
    transcription = transcription.strip()
    if len(transcription) < MIN_TRANSCRIPTION_LENGTH or len(transcription) > MAX_TRANSCRIPTION_LENGTH:
        return False
    
    # Check audio
    if audio_array is not None and sampling_rate is not None:
        # Calculate duration
        duration = len(audio_array) / sampling_rate
        if duration < MIN_AUDIO_DURATION or duration > MAX_AUDIO_DURATION:
            return False
        # Check audio is not all zeros or corrupted
        if np.all(audio_array == 0) or not np.isfinite(audio_array).all():
            return False
    
    # If audio_path is provided, check file exists (for local dataset)
    if audio_path is not None:
        if not os.path.exists(audio_path):
            return False
    
    return True

def load_local_common_voice_split(cv_path: str, split: str = "train") -> Optional[Dataset]:
    """
    Load a specific split from local Common Voice dataset.
    
    Args:
        cv_path: Path to the Common Voice language directory (e.g., /path/to/hi/)
        split: One of "train", "dev", "test", or "validated"
    
    Returns:
        HuggingFace Dataset or None if loading fails.
    """
    if not os.path.exists(cv_path):
        print(f"Local Common Voice path not found: {cv_path}")
        return None
    
    # Map split names to TSV files
    split_to_file = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "validation": "dev.tsv",
        "test": "test.tsv",
        "validated": "validated.tsv"
    }
    
    if split not in split_to_file:
        print(f"Unknown split: {split}. Using train.tsv")
        split = "train"
    
    tsv_file = split_to_file[split]
    tsv_path = os.path.join(cv_path, tsv_file)
    
    if not os.path.exists(tsv_path):
        print(f"TSV file not found: {tsv_path}")
        return None
    
    print(f"Loading local Common Voice from: {tsv_file} (split: {split})")
    
    try:
        # Read TSV file
        df = pd.read_csv(tsv_path, sep="\t")
        
        # Check required columns
        if "path" not in df.columns or "sentence" not in df.columns:
            print(f"TSV file missing required columns. Found: {df.columns.tolist()}")
            return None
        
        clips_dir = os.path.join(cv_path, "clips")
        if not os.path.exists(clips_dir):
            print(f"Clips directory not found: {clips_dir}")
            return None
        
        # Build dataset structure - OPTIMIZED: Don't load audio into memory
        # Only store paths and metadata, let Audio() handle loading lazily
        data = []
        valid_count = 0
        invalid_count = 0
        
        print(f"  Processing {len(df)} samples (lazy loading audio)...")
        
        for idx, row in df.iterrows():
            audio_filename = row["path"]
            transcription = str(row["sentence"]).strip()
            audio_path = os.path.join(clips_dir, audio_filename)
            
            # Quick validation - only check file exists and transcription
            if not is_valid_sample(audio_path=audio_path, transcription=transcription):
                invalid_count += 1
                continue
            
            # Store only path - Audio() will load lazily when needed
            # This prevents loading all audio files into memory at once
            data.append({
                "path": audio_path,
                "audio": audio_path,  # Store path only, not the array
                "sentence": transcription
            })
            valid_count += 1
            
            # Progress indicator for large datasets
            if (idx + 1) % 10000 == 0:
                print(f"    Processed {idx + 1}/{len(df)} samples...")
        
        if len(data) == 0:
            print(f"No valid samples found in {split} split")
            return None
        
        print(f"  Found {valid_count} valid samples, skipped {invalid_count} invalid samples")
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_list(data)
        
        # Cast audio column - this will load audio lazily when accessed
        # Audio() will handle resampling and loading on-demand
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        return dataset
        
    except Exception as e:
        print(f"Error loading local Common Voice {split} split: {e}")
        return None

print("Loading Hindi dataset for training...")
# Priority: Use local Mozilla Common Voice dataset if available
# Online datasets will only be loaded if local dataset fails or USE_ONLINE_DATASETS is set to "always"

# Load all available datasets (local + online)
train_datasets = []
dev_datasets = []

# 1. Try to load local Common Voice dataset splits
print("="*60)
print("Step 1: Loading local Common Voice dataset splits...")
print("="*60)

# Load train split
train_dataset = load_local_common_voice_split(LOCAL_CV_PATH, "train")
local_dataset_loaded = False

if train_dataset is not None:
    print(f"✅ Loaded train split: {len(train_dataset)} samples")
    train_datasets.append(("local_cv_train", train_dataset))
    local_dataset_loaded = True
    gc.collect()
else:
    print("⚠️  Train split not found, trying validated.tsv...")
    # Fallback to validated.tsv if train.tsv doesn't exist
    validated_dataset = load_local_common_voice_split(LOCAL_CV_PATH, "validated")
    if validated_dataset is not None:
        print(f"✅ Loaded validated dataset: {len(validated_dataset)} samples")
        train_datasets.append(("local_cv_validated", validated_dataset))
        local_dataset_loaded = True
        gc.collect()

# Load dev/validation split
dev_dataset = load_local_common_voice_split(LOCAL_CV_PATH, "dev")
if dev_dataset is not None:
    print(f"✅ Loaded dev split: {len(dev_dataset)} samples")
    dev_datasets.append(("local_cv_dev", dev_dataset))
    gc.collect()
else:
    print("⚠️  Dev split not found - will create validation split from training data")

if not local_dataset_loaded:
    print("⚠️  Local dataset not available, continuing with online datasets only")

# 2. Load online datasets (only if needed)
should_load_online = (
    USE_ONLINE_DATASETS == "always" or 
    (USE_ONLINE_DATASETS != "never" and not local_dataset_loaded)
)

if not should_load_online and local_dataset_loaded:
    print("\n" + "="*60)
    print("Step 2: Skipping online datasets")
    print("="*60)
    print("✅ Using local Common Voice dataset only")
    print(f"   Configuration: USE_ONLINE_DATASETS={USE_ONLINE_DATASETS}")
    print("   (Set USE_ONLINE_DATASETS='always' to load online datasets)")
    print("="*60)
else:
    print("\n" + "="*60)
    print("Step 2: Loading online datasets...")
    print("="*60)

dataset_configs = [
    # 1. Fixie.ai's Parquet version of Common Voice 17.0 (Try ignoring verification)
    {
        "name": "fixie-ai/common_voice_17_0",
        "config": "hi",
        "split": "train",
        "streaming": True,  # OPTIMIZED: Use streaming to avoid loading into memory
        "trust_remote_code": True,
        "verification_mode": "no_checks"
    },
    # 2. Common Voice 11.0 (Very stable older version)
    {
        "name": "mozilla-foundation/common_voice_11_0",
        "config": "hi",
        "split": "train",
        "streaming": True,  # OPTIMIZED: Use streaming
        "trust_remote_code": True
    },
    # 3. IndicVoices (Correct capitalization is important)
    {
        "name": "ai4bharat/IndicVoices",
        "config": "hi",
        "split": "train",
        "streaming": True,  # OPTIMIZED: Use streaming
        "trust_remote_code": True
    },
    # 4. Kathbath (Indian Accented Noisy Dataset)
    {
        "name": "ai4bharat/kathbath",
        "config": "hindi",
        "split": "train",
        "streaming": True,  # OPTIMIZED: Use streaming
        "trust_remote_code": True
    }
]

# Only load online datasets if should_load_online is True
if should_load_online:
    for config in dataset_configs:
        # Limit number of datasets to prevent memory issues
        if len(train_datasets) >= MAX_DATASETS_TO_LOAD:
            print(f"\n⚠️  Reached maximum dataset limit ({MAX_DATASETS_TO_LOAD}), skipping remaining datasets")
            break
        
        try:
            dataset_name = config["name"]
            lang_code = config["config"]
            streaming = config.get("streaming", False)
            trust_remote = config.get("trust_remote_code", True)
            verification_mode = config.get("verification_mode", None)
            
            print(f"\nTrying to load: {dataset_name} (language: {lang_code})...")
            
            load_kwargs = {
                "split": config["split"],
                "trust_remote_code": trust_remote
            }
            if streaming:
                load_kwargs["streaming"] = True
            if verification_mode:
                load_kwargs["verification_mode"] = verification_mode
            
            online_dataset = load_dataset(dataset_name, lang_code, **load_kwargs)
            
            # If streaming, convert to regular dataset with limit
            if streaming:
                print(f"Streaming mode: Collecting first {MAX_SAMPLES_PER_DATASET} samples...")
                # Collect samples from streaming dataset
                collected_samples = []
                for i, sample in enumerate(online_dataset):
                    if i >= MAX_SAMPLES_PER_DATASET:
                        break
                    collected_samples.append(sample)
                    if (i + 1) % 10000 == 0:
                        print(f"    Collected {i + 1}/{MAX_SAMPLES_PER_DATASET} samples...")
                
                # Create regular dataset from collected samples
                online_dataset = Dataset.from_list(collected_samples)
                del collected_samples  # Free memory
                gc.collect()
                
                # Cast audio column
                online_dataset = online_dataset.cast_column("audio", Audio(sampling_rate=16000))
            else:
                # Cast audio column
                online_dataset = online_dataset.cast_column("audio", Audio(sampling_rate=16000))
            
            print(f"✅ Successfully loaded: {dataset_name} ({len(online_dataset)} samples)")
            
            # Skip filtering - datasets are already quality-checked
            train_datasets.append((dataset_name, online_dataset))
            
            # Force garbage collection after each dataset
            gc.collect()
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {str(e)[:200]}...")
            continue
else:
    print("   (Skipped - using local dataset only)")

# 3. Combine datasets
print("\n" + "="*60)
print("Step 3: Combining datasets...")
print("="*60)

if len(train_datasets) == 0:
    print("\n" + "="*60)
    print("ERROR: Could not load any training dataset.")
    print("="*60)
    print("\nPossible solutions:")
    if USE_ONLINE_DATASETS == "never":
        print("1. Verify local dataset path: " + LOCAL_CV_PATH)
        print("2. Check that the directory contains train.tsv or validated.tsv")
        print("3. Ensure clips/ directory exists with audio files")
    else:
        print("1. Verify local dataset path: " + LOCAL_CV_PATH)
        print("2. Check that the directory contains train.tsv or validated.tsv")
        print("3. Ensure clips/ directory exists with audio files")
        print("4. Or set USE_ONLINE_DATASETS='always' to load online datasets")
        print("5. Accept terms of use for Common Voice:")
        print("   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
        print("6. Check if you're logged in: huggingface-cli login")
    raise RuntimeError("Failed to load any training dataset.")

# Concatenate training datasets
if len(train_datasets) > 1:
    print("Concatenating training datasets...")
    train_datasets_to_merge = [ds for _, ds in train_datasets]
    train_dataset = concatenate_datasets(train_datasets_to_merge)
    del train_datasets_to_merge
else:
    train_dataset = train_datasets[0][1]

print(f"✅ Training dataset: {len(train_dataset)} samples")
for name, ds in train_datasets:
    print(f"   - {name}: {len(ds)} samples")

# Handle validation dataset
if len(dev_datasets) > 0:
    if len(dev_datasets) > 1:
        print("Concatenating validation datasets...")
        dev_datasets_to_merge = [ds for _, ds in dev_datasets]
        dev_dataset = concatenate_datasets(dev_datasets_to_merge)
        del dev_datasets_to_merge
    else:
        dev_dataset = dev_datasets[0][1]
    
    print(f"✅ Validation dataset: {len(dev_dataset)} samples")
    for name, ds in dev_datasets:
        print(f"   - {name}: {len(ds)} samples")
else:
    # No dev split found - create validation split from training data
    print("⚠️  No dev split found - creating validation split from training data (90/10 split)...")
    split_result = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_result["train"]
    dev_dataset = split_result["test"]
    print(f"✅ Created validation split: {len(dev_dataset)} samples from training data")

gc.collect()

# 4. Shuffle datasets
print("\n" + "="*60)
print("Step 4: Shuffling datasets...")
print("="*60)

# Shuffle training dataset
train_dataset = train_dataset.shuffle(seed=42)
# Don't shuffle validation dataset - keep it consistent for evaluation

print(f"✅ Training dataset: {len(train_dataset)} samples (shuffled)")
print(f"✅ Validation dataset: {len(dev_dataset)} samples")
print("="*60)

# 2. Model and Processor - Load existing fine-tuned model
fine_tuned_model_id = "chaudharyritik1/whisper-hindi-v1"
print(f"Loading fine-tuned model from: {fine_tuned_model_id}")

# Load processor from the fine-tuned model
processor = WhisperProcessor.from_pretrained(fine_tuned_model_id, language="hindi", task="transcribe")

# 3. Preprocessing with Error Handling
def prepare_dataset(batch):
    """
    Preprocess dataset batch with error handling.
    Returns None for invalid samples (will be filtered out).
    """
    try:
        # Load and resample audio data to 16kHz
        audio = batch.get("audio")
        if audio is None:
            return None
        
        # Handle different audio formats
        audio_array = None
        sampling_rate = None
        
        if isinstance(audio, dict):
            # Audio already loaded as dict (from Audio() feature)
            audio_array = audio.get("array")
            sampling_rate = audio.get("sampling_rate")
        elif isinstance(audio, str):
            # Audio path - load it directly (Audio() feature should have decoded, but handle fallback)
            try:
                audio_array, sampling_rate = sf.read(audio)
                # Resample to 16kHz if needed
                if sampling_rate != 16000:
                    try:
                        from scipy import signal
                        num_samples = int(len(audio_array) * 16000 / sampling_rate)
                        audio_array = signal.resample(audio_array, num_samples)
                        sampling_rate = 16000
                    except ImportError:
                        # If scipy not available, assume Audio() already resampled
                        pass
            except Exception as e:
                return None
        else:
            return None
        
        # Basic validation - just check audio exists
        if audio_array is None or sampling_rate is None or len(audio_array) == 0:
            return None
        
        # Compute log-Mel input features from input audio array
        try:
            batch["input_features"] = processor.feature_extractor(
                audio_array, sampling_rate=sampling_rate
            ).input_features[0]
        except Exception as e:
            return None
        
        # Clear audio_array from memory immediately after processing
        del audio_array
        if "audio" in batch:
            del batch["audio"]
        
        # Encode target text to label ids
        # Common Voice uses 'sentence'
        text = None
        for col in ["sentence", "transcription", "text"]:
            if col in batch and batch[col] is not None:
                text = str(batch[col]).strip()
                break
        
        if text is None or len(text) == 0:
            # Try to find any text-like column
            text_cols = [col for col in batch.keys() if col not in ["audio", "input_features", "path"]]
            if text_cols:
                text = str(batch[text_cols[0]]).strip()
            else:
                return None
        
        # Basic validation - just check text exists
        if not text or len(text.strip()) == 0:
            return None
        
        # Tokenize
        try:
            batch["labels"] = processor.tokenizer(text).input_ids
            if len(batch["labels"]) == 0:
                return None
        except Exception as e:
            return None
        
        return batch
        
    except Exception as e:
        # Return None for any error - will be filtered out
        return None

# Apply preprocessing with memory optimization
print("\n" + "="*60)
print("Step 5: Preprocessing datasets...")
print("="*60)

# Preprocess dataset - handle both single examples and batches
def prepare_dataset_wrapper(examples):
    """Wrapper to handle both batched and non-batched inputs"""
    # Get audio list
    audio_list = examples.get("audio", [])
    if len(audio_list) == 0:
        return {"input_features": [], "labels": []}
    
    # Batched input - process each example
    results = []
    batch_size = len(audio_list) if isinstance(audio_list, list) else 1
    
    for i in range(batch_size):
        # Extract single example from batch
        example = {}
        for key in examples.keys():
            if isinstance(examples[key], list):
                example[key] = examples[key][i]
            else:
                example[key] = examples[key]
        
        result = prepare_dataset(example)
        if result is not None:
            results.append(result)
        
        # Periodic garbage collection for large batches
        if (i + 1) % 25 == 0:  # Every 25 samples
            import gc
            gc.collect()
    
    if len(results) == 0:
        return {"input_features": [], "labels": []}
    
    # Combine results
    combined = {}
    for key in results[0].keys():
        combined[key] = [r[key] for r in results]
    
    # Clear results to free memory
    del results
    import gc
    gc.collect()
    
    return combined

print("Preprocessing training dataset (this may take a while)...")
original_train_size = len(train_dataset)
train_dataset = train_dataset.map(
    prepare_dataset_wrapper,
    remove_columns=train_dataset.column_names,
    num_proc=1,
    batched=True,
    batch_size=50,  # REDUCED: Smaller batches to reduce memory
    writer_batch_size=200,  # REDUCED: Write more frequently to free memory
    desc="Preprocessing train",
    load_from_cache_file=False  # Don't cache to save disk space
)

# Skip filtering - datasets are already quality-checked
# train_dataset = train_dataset.filter(lambda x: len(x.get("input_features", [])) > 0 and len(x.get("labels", [])) > 0)

# Force garbage collection after training preprocessing
gc.collect()
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nPreprocessing validation dataset...")
original_dev_size = len(dev_dataset)

# Process validation in smaller chunks to avoid OOM
# Split validation into chunks of 500 samples
CHUNK_SIZE = 500
dev_chunks = []
total_chunks = (len(dev_dataset) + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"  Processing validation in {total_chunks} chunks of ~{CHUNK_SIZE} samples each...")

for chunk_idx in range(total_chunks):
    start_idx = chunk_idx * CHUNK_SIZE
    end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(dev_dataset))
    chunk = dev_dataset.select(range(start_idx, end_idx))
    
    print(f"  Processing validation chunk {chunk_idx + 1}/{total_chunks} ({start_idx}-{end_idx})...")
    
    processed_chunk = chunk.map(
        prepare_dataset_wrapper,
        remove_columns=chunk.column_names,
        num_proc=1,
        batched=True,
        batch_size=50,  # Small batches for memory efficiency
        writer_batch_size=200,  # Write frequently
        desc=f"Validation chunk {chunk_idx + 1}",
        load_from_cache_file=False
    )
    
    dev_chunks.append(processed_chunk)
    
    # Force garbage collection after each chunk
    del chunk
    del processed_chunk
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Concatenate all chunks
from datasets import concatenate_datasets
dev_dataset = concatenate_datasets(dev_chunks)
del dev_chunks
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Skip filtering - datasets are already quality-checked
# dev_dataset = dev_dataset.filter(lambda x: len(x.get("input_features", [])) > 0 and len(x.get("labels", [])) > 0)

# Force garbage collection after preprocessing
gc.collect()

print(f"\n✅ Final dataset sizes:")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Validation: {len(dev_dataset)} samples")

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
    gradient_accumulation_steps=4,  # Increased to stabilize training
    learning_rate=1e-5,  # Significantly lower learning rate for fine-tuning
    warmup_steps=50,
    max_steps=2000,
    lr_scheduler_type="linear",  # Linear decay
    gradient_checkpointing=True,
    bf16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,  # More frequent saves
    eval_steps=100,  # More frequent eval
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

# 8. Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("="*60)
print("Starting continued training on Common Voice Hindi dataset...")
print(f"Model: {fine_tuned_model_id}")
print(f"Dataset: Common Voice Hindi (train: {len(train_dataset)}, val: {len(dev_dataset)})")
print(f"Output directory: {training_args.output_dir}")
print("="*60)
trainer.train()

print("\n" + "="*60)
print("Training completed!")
print(f"Best model saved to: {training_args.output_dir}")
print("="*60)

# Final evaluation on validation set
print("\n" + "="*60)
print("Running final evaluation on validation set...")
print("="*60)
eval_results = trainer.evaluate()
print(f"\n📊 Final Evaluation Results:")
# Format metrics safely (handle missing keys)
wer = eval_results.get('eval_wer', None)
cer = eval_results.get('eval_cer', None)
precision = eval_results.get('eval_precision', None)
recall = eval_results.get('eval_recall', None)
f1 = eval_results.get('eval_f1', None)
accuracy = eval_results.get('eval_accuracy', None)

print(f"   WER (Word Error Rate):      {wer:.2f}%" if wer is not None else "   WER (Word Error Rate):      N/A")
print(f"   CER (Character Error Rate): {cer:.2f}%" if cer is not None else "   CER (Character Error Rate): N/A")
print(f"   Precision:                  {precision:.2f}%" if precision is not None else "   Precision:                  N/A")
print(f"   Recall:                     {recall:.2f}%" if recall is not None else "   Recall:                     N/A")
print(f"   F1-Score:                   {f1:.2f}%" if f1 is not None else "   F1-Score:                   N/A")
print(f"   Accuracy:                   {accuracy:.2f}%" if accuracy is not None else "   Accuracy:                   N/A")
print("="*60)

# Save evaluation results to file
results_file = os.path.join(training_args.output_dir, "final_evaluation_results.txt")
with open(results_file, "w") as f:
    f.write("Final Evaluation Results\n")
    f.write("="*60 + "\n")
    f.write(f"WER (Word Error Rate):      {wer:.2f}%\n" if wer is not None else "WER (Word Error Rate):      N/A\n")
    f.write(f"CER (Character Error Rate): {cer:.2f}%\n" if cer is not None else "CER (Character Error Rate): N/A\n")
    f.write(f"Precision:                  {precision:.2f}%\n" if precision is not None else "Precision:                  N/A\n")
    f.write(f"Recall:                     {recall:.2f}%\n" if recall is not None else "Recall:                     N/A\n")
    f.write(f"F1-Score:                   {f1:.2f}%\n" if f1 is not None else "F1-Score:                   N/A\n")
    f.write(f"Accuracy:                   {accuracy:.2f}%\n" if accuracy is not None else "Accuracy:                   N/A\n")
    f.write("="*60 + "\n")
    f.write(f"\nAll metrics:\n")
    for key, value in eval_results.items():
        f.write(f"  {key}: {value}\n")

print(f"\n✅ Evaluation results saved to: {results_file}")
print("="*60)


import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio, Dataset, DatasetDict
import evaluate
import os
import pandas as pd

# Configuration
MODEL_ID = "chaudharyritik1/whisper-hindi-v1"
DEFAULT_CV_PATH = "/Users/jaidratsingh/Downloads/cv-corpus-23.0-2025-09-05/hi"
CV_PATH = os.getenv("CV_PATH", DEFAULT_CV_PATH)

def load_local_common_voice(cv_path):
    """Load Common Voice dataset from local directory."""
    print(f"Loading test set from {cv_path}...")
    tsv_path = os.path.join(cv_path, "test.tsv")
    if not os.path.exists(tsv_path):
        raise ValueError(f"test.tsv not found in {cv_path}")
        
    df = pd.read_csv(tsv_path, sep="\t")
    df["audio"] = df["path"].apply(lambda x: os.path.join(cv_path, "clips", x))
    df = df.rename(columns={"sentence": "transcription"})
    
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

def evaluate_model():
    # 1. Load Dataset
    try:
        dataset = load_local_common_voice(CV_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Select a subset for faster evaluation if needed, or full test set
    # dataset = dataset.select(range(100)) # Uncomment for quick test
    print(f"Evaluating on {len(dataset)} samples...")

    # 2. Load Model and Processor
    print(f"Loading model: {MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="hindi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 3. Metric
    metric = evaluate.load("wer")

    # 4. Evaluation Loop
    print("Starting evaluation...")
    predictions = []
    references = []

    for i, item in enumerate(dataset):
        audio = item["audio"]
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        input_features = input_features.to(model.device).type(torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        reference = item["transcription"]

        predictions.append(transcription)
        references.append(reference)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(dataset)}")

    # 5. Compute WER
    wer = 100 * metric.compute(predictions=predictions, references=references)
    print(f"\n📊 Baseline WER for {MODEL_ID} on Common Voice Hindi: {wer:.2f}%")

if __name__ == "__main__":
    evaluate_model()

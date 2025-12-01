import os
import pandas as pd
from datasets import Dataset, Audio

def verify_setup():
    print("1. Verifying Dataset Access...")
    cv_path = "/Users/jaidratsingh/Downloads/cv-corpus-23.0-2025-09-05/hi"
    
    if not os.path.exists(cv_path):
        print(f"❌ Dataset path not found: {cv_path}")
        return

    print(f"✅ Found dataset directory: {cv_path}")
    
    # Check for train.tsv
    train_tsv = os.path.join(cv_path, "train.tsv")
    if not os.path.exists(train_tsv):
        print(f"❌ train.tsv not found in {cv_path}")
        return
        
    print(f"✅ Found train.tsv")
    
    try:
        # Load TSV
        df = pd.read_csv(train_tsv, sep="\t")
        print(f"✅ Loaded train.tsv. Rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create dataset from first 10 rows
        sample_df = df.head(10)
        # Add audio path
        sample_df["audio"] = sample_df["path"].apply(lambda x: os.path.join(cv_path, "clips", x))
        
        dataset = Dataset.from_pandas(sample_df)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        print("✅ Created HF Dataset from sample.")
        print(f"Sample audio path: {dataset[0]['audio']['path']}")
        
    except Exception as e:
        print(f"❌ Failed to process dataset: {e}")
        return

    print("\n2. Verifying Model Access...")
    model_id = "chaudharyritik/whisper-hindi-v1"
    
    try:
        processor = WhisperProcessor.from_pretrained(model_id, language="hindi", task="transcribe")
        print("✅ Processor loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load processor: {e}")

    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

if __name__ == "__main__":
    verify_setup()

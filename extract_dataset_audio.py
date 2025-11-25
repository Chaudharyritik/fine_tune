"""
Extract a sample audio file from the Google FLEURS (Hindi) dataset
"""

from datasets import load_dataset
import soundfile as sf
import os

print("Loading FLEURS dataset (validation split)...")
# Load validation split in non-streaming mode to ensure we get the file
dataset = load_dataset("google/fleurs", "hi_in", split="validation")

print("Extracting first sample...")
sample = dataset[0]
audio_data = sample["audio"]["array"]
sampling_rate = sample["audio"]["sampling_rate"]
text = sample["transcription"]

output_file = "fleurs_sample.wav"
sf.write(output_file, audio_data, sampling_rate)

print(f"\n✅ Saved sample to: {output_file}")
print(f"📝 Expected Transcription: {text}")
print("\nNow run:")
print(f"python3 test_local_model.py")
print(f"(Enter {output_file})")

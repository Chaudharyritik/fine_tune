"""
Test the fine-tuned model on Hindi audio
Run this after merging the model
"""

from transformers import pipeline
import torch

print("Loading fine-tuned model...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="./whisper-hindi-merged",
    torch_dtype=torch.float16,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Test with a sample (you'll need to provide your own audio file)
print("\nTesting model...")
print("Please provide path to a Hindi audio file:")
audio_path = input("> ")

if audio_path:
    result = pipe(audio_path)
    print(f"\nTranscription: {result['text']}")
else:
    print("\nNo audio file provided. Skipping test.")
    print("\nTo test later, run:")
    print("  python3 test_model.py")

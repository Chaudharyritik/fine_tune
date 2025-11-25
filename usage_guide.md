# How to Use Your Fine-Tuned Model in New Projects

Yes, you are correct! To use your model in a different project or computer, you primarily need the **`whisper-hindi-ct2`** folder.

## 1. What to Copy
Copy the entire folder `whisper-hindi-ct2` to your new project directory.

It should contain:
- `model.bin`
- `config.json`
- `vocabulary.json`
- `preprocessor_config.json` (optional but good to have)

## 2. Requirements
You need to install `faster-whisper` in your new environment:

```bash
pip install faster-whisper
```

## 3. Python Code
Here is the minimal code to load and use your model:

```python
from faster_whisper import WhisperModel

# Path to your copied folder
model_path = "./whisper-hindi-ct2"

# Load the model
# Run on CPU with int8 quantization (standard for Mac/CPU)
model = WhisperModel(model_path, device="cpu", compute_type="int8")

# Transcribe an audio file
segments, info = model.transcribe("your_audio.mp3", beam_size=5)

print(f"Detected language: {info.language}")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

## Summary
- **Portable**: Yes, the folder is self-contained.
- **No Internet Needed**: It will not download anything from Hugging Face.
- **Fast**: It uses CTranslate2 backend (via faster-whisper) which is highly optimized.

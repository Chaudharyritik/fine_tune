"""
Convert merged model to CTranslate2 format for faster inference
Run this on your Mac after downloading the merged model
"""

import subprocess
import sys

print("Converting model to CTranslate2 format...")
print("This will create an optimized model for faster inference on CPU/GPU")

try:
    subprocess.run([
        "ct2-transformers-converter",
        "--model", "./whisper-hindi-merged",
        "--output_dir", "./whisper-hindi-ct2",
        "--quantization", "int8"
    ], check=True)
    
    print("\n✅ Model converted successfully!")
    print("CTranslate2 model saved to: ./whisper-hindi-ct2")
    print("\nTo use the CTranslate2 model:")
    print("  from faster_whisper import WhisperModel")
    print("  model = WhisperModel('./whisper-hindi-ct2', device='cpu')")
    print("  segments, info = model.transcribe('audio.wav', language='hi')")
    
except FileNotFoundError:
    print("\n❌ Error: ct2-transformers-converter not found")
    print("Install it with: pip install ctranslate2")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"\n❌ Conversion failed: {e}")
    sys.exit(1)

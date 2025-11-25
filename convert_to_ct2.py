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
        "--model", "./whisper-hindi-v1",
        "--output_dir", "./whisper-hindi-ct2",
        "--quantization", "int8",
        "--force"
    ], check=True)
    
    print("\n✅ Model converted successfully!")
    
    # --- Auto-Fix Configuration ---
    import json
    import os
    
    config_path = "./whisper-hindi-ct2/config.json"
    if os.path.exists(config_path):
        print("🔧 Applying configuration fix...")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Ensure standard Whisper tokens are set
        # These are sometimes missing after conversion
        changes_made = False
        if "bos_token_id" not in config:
            config["bos_token_id"] = 50258
            changes_made = True
        if "eos_token_id" not in config:
            config["eos_token_id"] = 50257
            changes_made = True
        if "pad_token_id" not in config:
            config["pad_token_id"] = 50257
            changes_made = True
            
        if changes_made:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("  -> Config patched with missing token IDs")
    # ------------------------------

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

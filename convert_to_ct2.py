"""
Convert merged model to CTranslate2 format for faster inference
Run this on your Mac after downloading the merged model
"""

import subprocess
import sys
import os

# Configuration
INPUT_MODEL_DIR = "./whisper-hindi-merged"  # Default to merged model directory
OUTPUT_DIR = "./whisper-hindi-ct2"

# Allow override via command line argument
if len(sys.argv) > 1:
    INPUT_MODEL_DIR = sys.argv[1]
if len(sys.argv) > 2:
    OUTPUT_DIR = sys.argv[2]

# Check if input model exists
if not os.path.exists(INPUT_MODEL_DIR):
    print(f"\n❌ Error: Model directory not found: {INPUT_MODEL_DIR}")
    print(f"\nUsage: python {sys.argv[0]} [input_model_dir] [output_dir]")
    print(f"Example: python {sys.argv[0]} ./whisper-hindi-merged ./whisper-hindi-ct2")
    sys.exit(1)

print("Converting model to CTranslate2 format...")
print(f"Input model: {INPUT_MODEL_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print("This will create an optimized model for faster inference on CPU/GPU")

try:
    subprocess.run([
        "ct2-transformers-converter",
        "--model", INPUT_MODEL_DIR,
        "--output_dir", OUTPUT_DIR,
        "--quantization", "int8",
        "--force"
    ], check=True)
    
    print("\n✅ Model converted successfully!")
    
    # --- Auto-Fix Configuration ---
    import json
    
    config_path = os.path.join(OUTPUT_DIR, "config.json")
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

    print(f"CTranslate2 model saved to: {OUTPUT_DIR}")
    print("\nTo use the CTranslate2 model:")
    print("  from faster_whisper import WhisperModel")
    print(f"  model = WhisperModel('{OUTPUT_DIR}', device='cpu')")
    print("  segments, info = model.transcribe('audio.wav', language='hi')")
    
except FileNotFoundError:
    print("\n❌ Error: ct2-transformers-converter not found")
    print("Install it with: pip install ctranslate2")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"\n❌ Conversion failed: {e}")
    sys.exit(1)

import os
import shutil
from ctranslate2.converters import TransformersConverter

# Configuration
MODEL_PATH = "./whisper-hindi-cv-merged"
CT2_OUTPUT_DIR = "./whisper-hindi-cv-ct2"
QUANTIZATION = "float16" # Options: int8_float16, int8, float16

def convert_to_ct2():
    print(f"🚀 Converting model from '{MODEL_PATH}' to CTranslate2 format...")
    print(f"Output directory: {CT2_OUTPUT_DIR}")
    print(f"Quantization: {QUANTIZATION}")

    if os.path.exists(CT2_OUTPUT_DIR):
        print(f"⚠️ Output directory {CT2_OUTPUT_DIR} already exists. Removing it...")
        shutil.rmtree(CT2_OUTPUT_DIR)

    converter = TransformersConverter(
        model_or_model_dir=MODEL_PATH,
        copy_files=["tokenizer.json", "preprocessor_config.json"],
    )

    converter.convert(
        output_dir=CT2_OUTPUT_DIR,
        quantization=QUANTIZATION,
        force=True
    )

    print("\n" + "="*50)
    print("✅ Conversion Complete!")
    print("="*50)
    print(f"Model saved to: {CT2_OUTPUT_DIR}")
    print("\nTo use this model with faster-whisper:")
    print(f"model = WhisperModel('{CT2_OUTPUT_DIR}', device='cuda', compute_type='float16')")

if __name__ == "__main__":
    convert_to_ct2()

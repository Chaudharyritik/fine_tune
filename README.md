# Whisper Fine-Tuning (Hindi)

This repository contains a script to fine-tune `openai/whisper-large-v3-turbo` on the **Google FLEURS (Hindi)** dataset using **LoRA** and **8-bit quantization**.

## Requirements

- NVIDIA GPU (Required for 8-bit quantization)
- Python 3.8+

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Login to Hugging Face:**
    You need to be logged in to access models and datasets.
    ```bash
    huggingface-cli login
    ```

## Usage

Run the fine-tuning script:

```bash
python3 whisper_ft.py
```

## Configuration

- **Model:** `openai/whisper-large-v3-turbo`
- **Dataset:** `google/fleurs` (Hindi)
- **Method:** LoRA (Low-Rank Adaptation) + 8-bit Quantization
- **Metrics:** WER (Word Error Rate)

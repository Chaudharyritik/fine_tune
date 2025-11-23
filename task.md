# Task: Fine-Tune Whisper on YouTube Data

- [ ] Project Setup
    - [ ] Create `requirements.txt` with necessary dependencies (torch, transformers, datasets, yt-dlp, librosa, etc.)
    - [ ] Create `README.md` with project overview
- [ ] Data Acquisition Script
    - [ ] Implement YouTube audio downloader using `yt-dlp`
    - [ ] Implement subtitle/transcript downloader
- [ ] Data Preprocessing
    - [ ] Create script to align audio with transcripts
    - [ ] Format data for Hugging Face `datasets`
- [ ] Fine-Tuning Script
    - [ ] Setup Whisper model loading (PEFT/LoRA recommended)
    - [ ] Configure training arguments
    - [ ] Implement training loop
- [ ] Verification
    - [ ] Test inference with the fine-tuned model

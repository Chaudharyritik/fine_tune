"""
Real-time microphone transcription using the fine-tuned Whisper model.
Press Ctrl+C to stop recording and transcribe.
"""

import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import os

# Configuration
MODEL_PATH = "./whisper-hindi-ct2"
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

print("Loading model...")
model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")

def record_audio():
    """Record audio from microphone until user stops."""
    p = pyaudio.PyAudio()
    
    print("\n" + "="*50)
    print("🎤 Recording... Press Ctrl+C to stop and transcribe")
    print("="*50 + "\n")
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("\n⏹️  Recording stopped")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return frames

def save_audio(frames, filename):
    """Save recorded frames to WAV file."""
    p = pyaudio.PyAudio()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(audio_file):
    """Transcribe audio file using Whisper and measure performance."""
    import time
    import wave
    
    # Get audio duration
    with wave.open(audio_file, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        audio_duration = frames / float(rate)
    
    print(f"\n📊 Audio duration: {audio_duration:.2f}s")
    print("🔄 Transcribing...")
    
    # Measure transcription time
    start_time = time.time()
    segments, info = model.transcribe(audio_file, beam_size=5)
    
    # Collect all segments first to measure total time
    segment_list = list(segments)
    end_time = time.time()
    
    transcription_time = end_time - start_time
    real_time_factor = transcription_time / audio_duration if audio_duration > 0 else 0
    speed = audio_duration / transcription_time if transcription_time > 0 else 0
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   • Transcription time: {transcription_time:.2f}s")
    print(f"   • Real-time factor (RTF): {real_time_factor:.2f}x")
    print(f"   • Speed: {speed:.2f}x realtime")
    print(f"   • Latency: {transcription_time*1000:.0f}ms")
    
    print(f"\n🌍 Detected language: '{info.language}' (probability: {info.language_probability:.2f})")
    print("\n" + "="*50)
    print("📝 Transcription:")
    print("="*50)
    
    full_text = ""
    for segment in segment_list:
        print(segment.text)
        full_text += segment.text
    
    print("="*50)
    
    # Performance interpretation
    if real_time_factor < 0.5:
        perf_msg = "🚀 Excellent! Much faster than real-time"
    elif real_time_factor < 1.0:
        perf_msg = "✅ Good! Faster than real-time"
    elif real_time_factor < 1.5:
        perf_msg = "⚠️  Acceptable, slightly slower than real-time"
    else:
        perf_msg = "❌ Slow, significantly slower than real-time"
    
    print(f"\n{perf_msg}")
    print(f"(RTF < 1.0 means faster than real-time)\n")
    
    return full_text

def main():
    """Main loop for continuous recording and transcription."""
    print("🎙️  Real-time Whisper Transcription")
    print("Model: Hindi Fine-tuned Whisper v3 Turbo")
    print("\nTip: Speak clearly in Hindi or English")
    
    while True:
        try:
            # Record audio
            frames = record_audio()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_filename = tmp_file.name
            
            save_audio(frames, tmp_filename)
            
            # Transcribe
            transcribe_audio(tmp_filename)
            
            # Clean up
            os.remove(tmp_filename)
            
            # Ask if user wants to continue
            response = input("\n🔁 Record again? (y/n): ").strip().lower()
            if response != 'y':
                print("\n👋 Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    main()

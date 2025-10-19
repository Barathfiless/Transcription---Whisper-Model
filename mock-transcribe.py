import tkinter as tk
from tkinter import filedialog
from faster_whisper import WhisperModel
import torch
import os
from datetime import datetime

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def main():
    print("=" * 60)
    print("🎧 Audio File Transcription Tool")
    print("=" * 60)
    
    # Open file picker
    root = tk.Tk()
    root.withdraw()
    audio_path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[
            ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
            ("All files", "*.*")
        ]
    )
    
    # Check if file was selected
    if not audio_path:
        print("❌ No audio file selected.")
        return
    
    print(f"\n📁 Selected file: {os.path.basename(audio_path)}")
    print(f"📍 Path: {audio_path}\n")
    
    # Detect device
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
        print("🚀 Using GPU acceleration (CUDA)")
    else:
        device, compute_type = "cpu", "int8"
        print("💻 Using CPU")
    
    # Load model (use multilingual model)
    print("\n⏳ Loading Whisper model...")
    model = WhisperModel("small", device=device, compute_type=compute_type)
    print("✅ Model loaded successfully!\n")
    
    # Transcribe
    print("🎤 Transcribing audio...")
    print("-" * 60)
    
    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": float('inf'),
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400
            }
        )
        
        # Convert segments to list
        segments_list = list(segments)
        
        # Display audio info
        print(f"📊 Audio Duration: {format_timestamp(info.duration)}")
        print(f"🌐 Detected Language: {info.language.upper()} (confidence: {info.language_probability:.2%})")
        
        # Check if translation is needed
        needs_translation = info.language != 'en'
        
        if needs_translation:
            print(f"🔄 Translating from {info.language.upper()} to English...")
        
        # Show transcription header
        print("🎧 Transcription Result:")
        print("=" * 60)
        
        full_text = []
        
        # Display original transcription with accurate timestamps
        for segment in segments_list:
            # Round to nearest second for cleaner display
            start_seconds = round(segment.start)
            start_time = format_timestamp(start_seconds)
            text = segment.text.strip()
            
            print(f"[{start_time}] {text}")
            full_text.append(f"[{start_time}] {text}")
        
        print("=" * 60)
        
        # If not English, show English translation
        full_text_english = []
        if needs_translation:
            print("🌍 English Translation:")
            print("=" * 60)
            
            segments_en, _ = model.transcribe(
                audio_path,
                task="translate",
                beam_size=5,
                vad_filter=True,
                temperature=0.0,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "max_speech_duration_s": float('inf'),
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 400
                }
            )
            
            for segment in segments_en:
                start_seconds = round(segment.start)
                start_time = format_timestamp(start_seconds)
                text_en = segment.text.strip()
                
                print(f"[{start_time}] {text_en}")
                full_text_english.append(f"[{start_time}] {text_en}")
            
            print("=" * 60)
        
        print(f"✅ Transcription complete!")
        print(f"📝 Total segments: {len(full_text)}")
        
        # Option to save
        save_option = input("\n💾 Save transcription to file? (y/n): ").lower()
        if save_option == 'y':
            output_path = audio_path.rsplit('.', 1)[0] + '_transcription.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcription of: {os.path.basename(audio_path)}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {format_timestamp(info.duration)}\n")
                f.write(f"Language: {info.language.upper()}\n")
                f.write("=" * 60 + "\n\n")
                
                # Original transcription
                f.write("Original Transcription:\n")
                f.write("-" * 60 + "\n")
                f.write("\n".join(full_text))
                
                # English translation if available
                if full_text_english:
                    f.write("\n\n" + "=" * 60 + "\n\n")
                    f.write("English Translation:\n")
                    f.write("-" * 60 + "\n")
                    f.write("\n".join(full_text_english))
                
            print(f"✅ Saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()


# import tkinter as tk 
# from tkinter import filedialog 
# from faster_whisper import WhisperModel 
# import torch

# # Open file picker
# root = tk.Tk()
# root.withdraw()
# audio_path = filedialog.askopenfilename(title="Select an audio file", filetypes=[("Audio files", "*.mp3 *.wav *.m4a")])

# # Checking Audio Path
# if not audio_path:
#     raise FileNotFoundError("No audio file selected.")

# if torch.cuda.is_available():
#     device, compute_type = "cuda", "float16"
# else:
#     device, compute_type = "cpu", "int8"

# model = WhisperModel("small.en", device=device, compute_type=compute_type)

# segments, _ = model.transcribe(audio_path, language="en", beam_size=5)

# print("\n🎧 Transcription Result:\n" + "-"*40)
# for segment in segments:
#     print(segment.text)
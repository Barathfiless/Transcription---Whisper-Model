import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import sys
from datetime import datetime

# ----------------- SETTINGS -----------------
SAMPLERATE = 16000
BLOCK_DURATION = 0.1
CHANNELS = 1

# Voice Activity Detection settings
SPEECH_THRESHOLD = 0.01
SILENCE_DURATION = 1.5
MIN_SPEECH_DURATION = 0.5

FRAMES_PER_BLOCK = int(SAMPLERATE * BLOCK_DURATION)

audio_queue = queue.Queue()

# ----------------- MODEL -----------------
print("Loading Whisper model...")
model = WhisperModel("small.en", device="cpu", compute_type="int8")
print("Model loaded successfully!\n")

# ----------------- AUDIO CALLBACK -----------------
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

# ----------------- VOICE DETECTION -----------------
def calculate_energy(audio_chunk):
    return np.sqrt(np.mean(audio_chunk**2))

# ----------------- FIND A WORKING MIC -----------------
def find_working_input():
    devices = sd.query_devices()
    default_idx = sd.default.device[0]
    try:
        with sd.InputStream(device=default_idx, channels=1, samplerate=SAMPLERATE, blocksize=FRAMES_PER_BLOCK):
            return default_idx
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

DEVICE_INDEX = find_working_input()

# ----------------- RECORDER -----------------
def recorder():
    try:
        with sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            callback=audio_callback,
            blocksize=FRAMES_PER_BLOCK,
            device=DEVICE_INDEX
        ):
            while True:
                sd.sleep(100)
    except Exception as e:
        print(f"Recorder error: {e}")
        sys.exit(1)

# ----------------- TRANSCRIBER -----------------
def transcriber():
    speech_buffer = []
    is_speaking = False
    silence_start = None
    
    print("Transcribing...")
    print("-" * 60)
    
    try:
        while True:
            try:
                block = audio_queue.get(timeout=0.1)
                block_flat = block.flatten().astype(np.float32)
                energy = calculate_energy(block_flat)
                
                # Speech detection
                if energy > SPEECH_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        silence_start = None
                    
                    speech_buffer.append(block_flat)
                    
                else:
                    # Silence detected
                    if is_speaking:
                        speech_buffer.append(block_flat)
                        
                        if silence_start is None:
                            silence_start = datetime.now()
                        
                        silence_duration = (datetime.now() - silence_start).total_seconds()
                        
                        # Check if silence duration threshold is reached
                        if silence_duration >= SILENCE_DURATION:
                            # Calculate total speech duration
                            total_duration = len(speech_buffer) * BLOCK_DURATION
                            
                            if total_duration >= MIN_SPEECH_DURATION:
                                # Concatenate and transcribe
                                full_audio = np.concatenate(speech_buffer)
                                
                                try:
                                    segments, info = model.transcribe(
                                        full_audio,
                                        language="en",
                                        beam_size=5,
                                        vad_filter=True,
                                        vad_parameters=dict(
                                            min_silence_duration_ms=500
                                        )
                                    )
                                    
                                    segments_list = list(segments)
                                    text = " ".join([s.text for s in segments_list]).strip()
                                    
                                    if text:
                                        timestamp = datetime.now().strftime("%I.%M%p")
                                        print(f"[{timestamp}] - {text}")
                                        
                                except Exception as e:
                                    print(f"Error: {e}")
                            
                            # Reset
                            speech_buffer = []
                            is_speaking = False
                            silence_start = None
                    
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("Transcription stopped.")

# ----------------- MAIN -----------------
if __name__ == "__main__":
    threading.Thread(target=recorder, daemon=True).start()
    transcriber()
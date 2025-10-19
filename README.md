# Transcription---Whisper-Model

# 🎤 Real-time Speech Transcription

A Python-based real-time speech-to-text transcription tool using Faster Whisper. Captures audio from your microphone and transcribes it with timestamps.

## ✨ Features

- 🎙️ Real-time speech transcription
- 🔊 Automatic microphone detection
- 🔇 Smart silence detection (transcribes complete sentences)
- ⏰ Timestamped output
- 🚀 Fast processing with Faster Whisper
- 💻 Runs entirely on CPU (no GPU required)
- 🌐 English language support

## 📋 Requirements

- Python 3.8 or higher
- Working microphone
- ~500MB disk space for the model

## 🚀 Installation

1. **Clone this repository:**
```bash
git clone https://github.com/Barathfiless/Transcription---Whisper-Model.git
cd Transcription---Whisper-Model
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```
---

## 💡 Usage

### 1. Real-time Transcription

Transcribe speech from your microphone in real-time:

```bash
python realtime-transcribe.py
```

The program will:
1. Load the Whisper model (first run downloads ~500MB)
2. Start listening to your microphone
3. Transcribe speech after detecting 1.5 seconds of silence
4. Display transcriptions with timestamps

**Example Output:**
```
============================================================
🎤 Real-time Speech Transcription
============================================================

⏳ Loading Whisper model...
✅ Model loaded successfully!

🎙️  Using microphone: Microphone Array (Intel® Smart Sound Technology)

============================================================
🎧 Transcribing... (Press Ctrl+C to stop)
============================================================

[12.32PM] - Hello this is a test of the transcription system
[12.34PM] - It works really well for real time speech recognition
[12.35PM] - Press control C to stop the program
```

Press `Ctrl+C` to stop transcription.

---


### 2. Mock Transcription

Transcribe pre-recorded audio files:

```bash
python mock-transcribe.py
```

Features:
- 📁 File picker dialog to select audio files
- 🚀 Automatic GPU/CPU detection
- ⏱️ Timestamped output for each segment
- 💾 Option to save transcription to text file
- 📊 Audio duration and language detection

**Supported formats:** MP3, WAV, M4A, FLAC, OGG

**Example Output:**
```
============================================================
🎧 Audio File Transcription Tool
============================================================

📁 Selected file: interview.mp3
📍 Path: C:/Users/Documents/interview.mp3

💻 Using CPU

⏳ Loading Whisper model...
✅ Model loaded successfully!

🎤 Transcribing audio...
------------------------------------------------------------

📊 Audio Duration: 03:45
🌐 Language: en (confidence: 99.87%)

🎧 Transcription Result:
============================================================

[00:00] Welcome to today's interview
[00:15] Let's start with your background
[00:32] I have been working in tech for five years
...

============================================================
✅ Transcription complete!
📝 Total segments: 24

💾 Save transcription to file? (y/n):
```
---

## ⚙️ Configuration

Adjust these parameters in `realtime-transcribe.py`:

```python
SPEECH_THRESHOLD = 0.01      # Energy level to detect speech (lower = more sensitive)
SILENCE_DURATION = 1.5       # Seconds of silence before transcribing
MIN_SPEECH_DURATION = 0.5    # Minimum speech duration to process (seconds)
```
---

### Fine-tuning Sensitivity


|                Issue                      |                   Solution                      |
|-------------------------------------------|-------------------------------------------------|
| Too sensitive (picks up background noise) | Increase `SPEECH_THRESHOLD` to `0.02` or `0.03` |
| Missing quiet speech                      | Decrease `SPEECH_THRESHOLD` to `0.005`          |
| Cutting off sentences                     | Increase `SILENCE_DURATION` to `2.0` or `2.5`   |
| Transcribing too quickly                  | Increase `SILENCE_DURATION`                     |

---

## 🔧 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Capture   │ ──> │    Voice     │ ──> │   Buffer    │ ──> │  Transcribe  │
│    Audio    │     │   Activity   │     │    Audio    │     │   & Output   │
│  (16kHz)    │     │  Detection   │     │   Chunks    │     │  w/Timestamp │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

1. **Audio Capture**: Continuously captures audio from microphone at 16kHz
2. **Voice Activity Detection**: Monitors audio energy to detect speech
3. **Buffering**: Accumulates audio while you're speaking
4. **Silence Detection**: Waits for silence after speech ends
5. **Transcription**: Processes complete audio using Faster Whisper
6. **Output**: Displays transcribed text with timestamp

---

## 🐛 Troubleshooting

### No transcription appearing
- ✅ Check microphone volume in system settings
- ✅ Ensure correct microphone is set as default
- ✅ Lower `SPEECH_THRESHOLD` value
- ✅ Speak louder or closer to microphone

### Transcription cutting off mid-sentence
- ✅ Increase `SILENCE_DURATION` to wait longer

### Background noise being transcribed
- ✅ Increase `SPEECH_THRESHOLD` to reduce sensitivity
- ✅ Use better quality microphone
- ✅ Reduce ambient noise

### Model loading errors
- ✅ Ensure stable internet connection (first run)
- ✅ Check available disk space (~500MB needed)
- ✅ Try deleting `~/.cache/whisper/` and rerunning

---

## 🎯 Model Options

The script uses `small.en` model by default. Available models:


| Model      | Size   | Speed   | Accuracy | Use Case      |
|------------|--------|---------|----------|---------------|
| `tiny.en`  | ~75MB  | Fastest | Good     | Quick testing |
| `base.en`  | ~150MB | Fast    | Better   | Balanced      |
| `small.en` | ~500MB | Medium  | Great    | **Default**   |
| `medium.en`| ~1.5GB | Slow    | Best     | High accuracy |


Change in code:
```python
model = WhisperModel("base.en", device="cpu", compute_type="int8")
```

**Large-v2 Model on GPU**

-----------------------------------------------------------------------------------------------
| **Implementation**              | **Precision** | **Beam size** | **Time** | **VRAM Usage** |
| ------------------------------- | ------------- | ------------- | -------- | -------------- |
| openai/whisper                  | fp16          | 5             | 2m23s    | 4708 MB        |
| whisper.cpp (Flash Attention)   | fp16          | 5             | 1m05s    | 4127 MB        |
| transformers (SDPA)             | fp16          | 5             | 1m52s    | 4960 MB        |
| faster-whisper                  | fp16          | 5             | 1m03s    | 4525 MB        |
| faster-whisper *(batch_size=8)* | fp16          | 5             | 17s      | 6090 MB        |
| faster-whisper                  | int8          | 5             | 59s      | 2926 MB        |
| faster-whisper *(batch_size=8)* | int8          | 5             | 16s      | 4500 MB        |
-----------------------------------------------------------------------------------------------

**distil-whisper-large-v3 Model on GPU**

---------------------------------------------------------------------------------------------------------
| **Implementation**                    | **Precision** | **Beam size** | **Time** | **YT Commons WER** |
| ------------------------------------- | ------------- | ------------- | -------- | ------------------ |
| transformers (SDPA) *(batch_size=16)* | fp16          | 5             | 46m12s   | 14.801             |
| faster-whisper *(batch_size=16)*      | fp16          | 5             | 25m50s   | 13.527             |
---------------------------------------------------------------------------------------------------------

**Small Model on CPU**

----------------------------------------------------------------------------------------------
| **Implementation**              | **Precision** | **Beam size** | **Time** | **RAM Usage** |
| ------------------------------- | ------------- | ------------- | -------- | ------------- |
| openai/whisper                  | fp32          | 5             | 6m58s    | 2335 MB       |
| whisper.cpp                     | fp32          | 5             | 2m05s    | 1049 MB       |
| whisper.cpp (OpenVINO)          | fp32          | 5             | 1m45s    | 1642 MB       |
| faster-whisper                  | fp32          | 5             | 2m37s    | 2257 MB       |
| faster-whisper *(batch_size=8)* | fp32          | 5             | 1m06s    | 4230 MB       |
| faster-whisper                  | int8          | 5             | 1m42s    | 1477 MB       |
| faster-whisper *(batch_size=8)* | int8          | 5             | 51s      | 3608 MB       |
----------------------------------------------------------------------------------------------

---

## ⚡ Performance Tips

- 🔸 Close heavy applications for better performance
- 🔸 Use `compute_type="int8"` for faster CPU inference
- 🔸 Consider `tiny.en` or `base.en` on slower machines
- 🔸 Ensure microphone is not being used by other apps

---

## 📁 Project Structure

```
Transcription---Whisper-Model/
├── realtime-transcribe.py    # Real-time microphone transcription
├── mock-transcribe.py         # Audio file transcription with GUI
├── README.md                  # Documentation
├── requirements.txt           # Python dependencies
└── .gitignore                 # Git ignore file
```

---

### Script Comparison


| Feature        | realtime-transcribe.py   | mock-transcribe.py             |
|----------------|--------------------------|--------------------------------|
| **Input**      | Microphone (live)        | Audio file                     |
| **Output**     | Console with timestamps  | Console + optional file save   |
| **UI**         | Command-line             | File picker dialog             |
| **Use Case**   | Live meetings, dictation | Pre-recorded audio, interviews |
| **Processing** | Continuous streaming     | Single file processing         |

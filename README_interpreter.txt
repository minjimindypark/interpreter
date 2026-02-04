🎧 Real-time PC Audio Translator
A Python tool that captures internal computer audio (YouTube, Netflix, Zoom meetings, etc.) in real-time, transcribes English speech, and translates it into Korean subtitles directly in your terminal.

Unlike standard dictation tools, this captures System Audio (Loopback), meaning it works perfectly with headphones without needing a microphone input.

✨ Key Features
System Audio Capture: Listens to what your computer is playing, not your microphone.

Faster-Whisper: Powered by an optimized version of OpenAI's Whisper model for fast CPU inference (Default: tiny.en).

Real-time Translation: Instantly translates transcribed English into Korean using deep_translator.

Clean UI (Refresh Mode): Refreshes the terminal screen (cls/clear) continuously to prevent duplicate text or scrolling issues, ensuring a clean subtitle experience.

VAD (Voice Activity Detection): Automatically ignores silence to save processing power and keep the output clean.

🛠️ Prerequisites
This project is optimized for Windows environments (specifically for WASAPI Loopback support).

Python 3.8 or higher

FFmpeg (Required for audio processing in Whisper)

📦 Installation
Clone this repository or download the interpreter.py file.

Install the required Python libraries:

Bash
pip install faster-whisper deep-translator pyaudiowpatch scipy numpy
Note: pyaudiowpatch is a fork of PyAudio that supports WASAPI Loopback (system audio capture) on Windows.

🚀 Usage
Run the script:

Bash
python interpreter.py
Wait for the message ✅ Connected: ... in the console.

Play any content containing English speech (YouTube, movies, podcasts) on your PC.

The terminal will display the live transcription and translation.

⚙️ Configuration
You can adjust the settings at the top of interpreter.py to fit your needs:

Python
# ==================================================================
# 🔧 System Settings
# ==================================================================
MODEL_SIZE = "tiny.en"     # Model size (tiny.en, base.en, small.en, etc.)
                           # Use 'base.en' if you have a decent CPU.
TARGET_SR = 16000          # Target Sample Rate (Do not change)
VAD_THRESHOLD = 0.015      # Noise threshold (Increase if background noise is detected)
UPDATE_INTERVAL = 0.3      # Screen refresh rate (Lower = more responsive)
SILENCE_TIMEOUT = 0.8      # Seconds of silence required to finalize a sentence
# ==================================================================
⚠️ Troubleshooting
Error: ❌ No device found / Device not found

Ensure your speakers or headphones are connected and set as the Default Output Device in Windows Sound Settings.

This tool relies on WASAPI Loopback; ensure no exclusive mode application is blocking audio access.

Translation is too slow:

Ensure MODEL_SIZE is set to "tiny.en".

You can adjust cpu_threads=4 in the code to match your processor's capabilities.


Libraries Used
Faster-Whisper

Deep Translator

PyAudioWPatch
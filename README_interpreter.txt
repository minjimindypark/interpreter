🚀 Real-Time AI Translator (WASAPI Loopback)
A high-performance Python tool that captures Windows system audio (WASAPI Loopback) in real-time, transcribes English speech, and translates it into Korean instantly.

This version is highly optimized for speed (low latency), supports mono headsets (like Jabra), and runs entirely in the console without saving log files.

✨ Key Features
Real-Time Audio Capture: Uses pyaudiowpatch to capture system audio (Loopback) directly.

Ultra-Fast STT: Powered by faster-whisper (default: tiny model) for low-latency speech recognition on CPUs.

Instant Translation: Uses Google Translate API for English-to-Korean translation.

Performance Optimizations:

High-Speed Resampling: Uses numpy slicing instead of interpolation (100x faster).

Queue Management: Automatically drops old data to prevent latency buildup.

Hallucination Filter: Prevents Whisper from generating random text during silence.

Robust Device Support: Includes a fix for the PaErrorCode -9998 error, enabling support for 1-channel (Mono) headsets like Jabra in Loopback mode.

📋 Prerequisites
Before running the script, ensure you have the following installed:

Python 3.10 or higher (3.11 recommended)

FFmpeg: Required for audio processing. Download FFmpeg and add it to your system PATH.

Microsoft Visual C++ Redistributable: Required for PyTorch/Whisper. Download here.

🛠️ Installation
Clone this repository or download the script.

Install the required Python packages:

Bash
pip install pyaudiowpatch faster-whisper deep-translator numpy
Note on Models: Unlike previous versions that required manual dictionary downloads (e.g., Argos Translate), this version uses the Google Translate API. However, the Whisper model (approx. 70MB for tiny) will be downloaded automatically the first time you run the script.

🚀 Usage
Run the script directly with Python:

Bash
interpreter.py
Stop: Press Ctrl + C to exit.

⚙️ Configuration
You can adjust the settings at the top of the app_manual3_nolog.py file:

Python
# Model Size: tiny < base < small
# 'tiny' is the fastest. 'base' offers slightly better accuracy but is slower.
WHISPER_MODEL = "tiny" 

# Chunk Size (Seconds)
# Recommended: 2.0 ~ 4.0. 
# Too short (<1.5s) may cut off sentences; too long (>5s) increases latency.
CHUNK_SECONDS = 4.0    
⚠️ Troubleshooting
Q. Error: Error opening InputStream: Invalid number of channels [PaErrorCode -9998]

This script includes a patch for Mono headsets. However, if the error persists, check your Windows Sound Settings:

Go to Sound Control Panel -> Playback tab.

Right-click your device -> Properties -> Advanced tab.

Uncheck both boxes under "Exclusive Mode" (Allow applications to take exclusive control...).

Q. The translation is delayed.

Ensure WHISPER_MODEL is set to "tiny".

Delays may occur depending on your internet connection speed (due to Google Translate API).


📜 License
This project is intended for educational and research purposes.
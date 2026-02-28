🎧 Real-time PC Audio Translator (Live Caption V3)
A Python tool that captures internal computer audio (YouTube, Netflix, Zoom meetings, etc.) in real-time, transcribes English speech, and translates it into Korean subtitles directly in your terminal — at live caption quality.

Unlike standard dictation tools, this captures System Audio (Loopback), meaning it works perfectly with headphones without needing a microphone input.

✨ Key Features
System Audio Capture: Listens to what your computer is playing, not your microphone.

Faster-Whisper: Powered by an optimized version of OpenAI's Whisper model for fast CPU inference (Default: tiny.en).

Real-time Translation (Async): Partial results trigger background translation; final sentences use a precise synchronous call. The main display is never blocked by network requests.

Prefix-lock Stabilization: Common word prefixes between updates are locked so only the tail of the sentence rewrites — no more "shaky subtitle" effect.

Live/Final 2-Track Transcription:
  - Live Track (every 0.3 s): beam_size=1, vad_filter=True for instant partial captions (shown in gray).
  - Final Track (sentence end): beam_size=3 for accurate re-transcription (shown in bold white).

VAD Pre-roll Buffer: The last 0.3 s of audio before speech is detected is prepended to each utterance, preventing the first word from being clipped.

Flicker-free ANSI UI: Uses ANSI cursor control (\033[H\033[J) instead of cls/clear — no screen flash. Fully compatible with Windows 10+ terminals (VT100 mode is enabled automatically).

🛠️ Prerequisites
This project is optimized for Windows 10+ environments (WASAPI Loopback support).

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

The terminal will display the live transcription (gray) and finalized translation (bold white).

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
UPDATE_INTERVAL = 0.3      # Live caption refresh rate in seconds (Lower = more responsive)
SILENCE_TIMEOUT = 0.6      # Seconds of silence to finalize a sentence (faster than before)
PREROLL_CHUNKS = 3         # Pre-roll buffer size (3 × 0.1 s = 0.3 s) to prevent word clipping
# ==================================================================

⚠️ Troubleshooting
Error: ❌ No device found / Device not found

Ensure your speakers or headphones are connected and set as the Default Output Device in Windows Sound Settings.

This tool relies on WASAPI Loopback; ensure no exclusive mode application is blocking audio access.

Translation is too slow:

Ensure MODEL_SIZE is set to "tiny.en".

You can adjust cpu_threads=4 in the code to match your processor's capabilities.

ANSI colors not showing on Windows:

The script automatically enables VT100 mode via ctypes on Windows 10+. If colors still do not appear, run the terminal as Administrator or use Windows Terminal instead of cmd.exe.

Libraries Used
Faster-Whisper

Deep Translator

PyAudioWPatch
OpenAI Transcribe Drop Folder

Overview
- Drop audio files into input/
- The script transcribes with gpt-4o-transcribe
- A new timestamped folder is created inside output/ for each file
- Each bundle includes the audio, raw transcript, and edited transcript (if enabled)
- Errors go to failed/

Works on
- macOS, Windows, and Linux (Python 3.9+ recommended)

Folder layout
- input/
- output/
- failed/

Setup
1) Create a virtualenv (optional) and install deps:
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt

2) Create a .env file in this folder (see .env.example):
   OPENAI_API_KEY="..."
   MODEL=gpt-4o-transcribe
   RESPONSE_FORMAT=text
   PROMPT="Transcribe verbatim, keep fillers and corrections, and preserve technical terms exactly."
   EDIT_MODEL=gpt-5.2-2025-12-11
   EDIT_PROMPT="Rewrite the transcript into clear, professional English while preserving meaning. Fix grammar and punctuation, remove filler words, and keep all technical details unchanged. If unclear, mark [unclear]."

Run
- Continuous watch:
  python3 watch_transcribe.py

- One-time batch:
  python3 watch_transcribe.py --once

Notes
- Supported audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
- Files over 25 MB must be split before upload
- Each processed file creates a new bundle folder under output/

Example
  MODEL=gpt-4o-transcribe python3 watch_transcribe.py

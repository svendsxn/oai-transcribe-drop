OpenAI Transcribe Drop Folder

What it does
- Drop audio files into the input folder
- The script transcribes with gpt-4o-transcribe
- A new timestamped folder is created inside output for each file
- That folder contains the audio, raw transcript, and edited transcript (if enabled)
- Errors go to failed

Folders
- input
- output
- failed

Setup
1) Create a virtualenv (optional) and install deps:
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt

2) Create a .env file in this folder (see .env.example):
   OPENAI_API_KEY="..."

Run
- Continuous watch:
  python3 watch_transcribe.py

- One-time batch:
  python3 watch_transcribe.py --once

Options
- MODEL: default gpt-4o-transcribe
- RESPONSE_FORMAT: default text
- PROMPT: optional prompt string
- EDIT_MODEL: optional editor model (e.g. gpt-5.2-2025-12-11)
- EDIT_PROMPT: optional editor prompt; when set, edited output is written to the same timestamped folder
- POLL_SECONDS: polling interval (default 2)
- MIN_AGE_SECONDS: min age to consider file ready (default 1)

Example
  MODEL=gpt-4o-transcribe python3 watch_transcribe.py

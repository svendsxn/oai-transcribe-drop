OpenAI Transcribe Drop Folder

Overview
- Drop audio files into input/
- The script transcribes with gpt-4o-transcribe
- You can switch to gpt-4o-transcribe-diarize for multi-speaker meetings
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
   EDIT_PROMPT="Create a concise meeting summary with sections: Summary, Decisions, Action Items, Risks, and Open Questions. Action Items must be a bullet list with owner (if known) and due date (if mentioned). If something is unclear, mark it as [unclear]. Our company name is MODU Valves and we sell industrial valves - normalize references to the company and products accordingly."
   FORMAT=meeting_summary
   FORMATS=meeting_summary,action_items,internal_email,sales_email,linkedin_post,blog_post,sop,knowledge_piece
   TONE=professional
   LENGTH=medium
   AUDIENCE=internal
   COMPANY_NAME="MODU Valves"
   COMPANY_CONTEXT="We sell industrial valves."
   CORRECTIONS_FILE=corrections.txt
   NAMES_FILE=
   USE_CORRECTIONS=true

Run (simple)
First time (fast setup)
1) cd to the folder:
   cd /Users/gustavsvendsen/oai-transcribe-drop
2) Install deps:
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
3) Run once:
   python3 watch_transcribe.py --once

Already set up
1) cd /Users/gustavsvendsen/oai-transcribe-drop
2) (if you use venv) . .venv/bin/activate
3) Run:
   python3 watch_transcribe.py --once

Continuous watch (keep running)
  python3 watch_transcribe.py

If installed as a command (e.g., `oai-transcribe-drop`)
- Run once:
  oai-transcribe-drop --once
- Continuous watch:
  oai-transcribe-drop

Diarized meetings (multi-speaker)
- Set MODEL to gpt-4o-transcribe-diarize
- Set RESPONSE_FORMAT to diarized_json (auto-selected if not set)
- Optional: CHUNKING_STRATEGY=auto (recommended for long files)
- PROMPT is ignored for diarization models
- If EDIT_PROMPT is not set, a default meeting-summary prompt is used for diarization
- Outputs:
  - <name>.raw.txt (if the API returns text)
  - <name>.diarized.json (full diarized response)
  - <name>.diarized.txt (speaker-labeled text)
  - <name>.edited.txt (single format) or <name>.<format>.txt (multiple formats)

Formats
- meeting_summary
- action_items
- internal_email
- sales_email
- linkedin_post
- blog_post
- sop
- knowledge_piece

Corrections and names
- Add corrections and name spellings to corrections.txt (see the NAMES and CORRECTIONS sections)
- JSON corrections are also supported (key: transcription_corrections)

Notes
- Supported audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
- Files over 25 MB must be split before upload
- Each processed file creates a new bundle folder under output/

Example
  MODEL=gpt-4o-transcribe python3 watch_transcribe.py

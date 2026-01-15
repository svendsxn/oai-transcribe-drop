#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}


def is_file_ready(path: Path, min_age_seconds: float) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    if stat.st_size == 0:
        return False
    age = time.time() - stat.st_mtime
    return age >= min_age_seconds


def transcribe_file(client: OpenAI, path: Path, model: str, response_format: str, prompt: str | None) -> str:
    with path.open("rb") as audio_file:
        kwargs = {
            "model": model,
            "file": audio_file,
            "response_format": response_format,
        }
        if prompt:
            kwargs["prompt"] = prompt
        transcription = client.audio.transcriptions.create(**kwargs)
    if hasattr(transcription, "text"):
        return transcription.text
    return str(transcription)


def edit_transcript(client: OpenAI, text: str, model: str, prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    return response.output_text


def write_output(output_dir: Path, name: str, text: str) -> Path:
    output_path = output_dir / name
    output_path.write_text(text, encoding="utf-8")
    return output_path


def move_to(dir_path: Path, file_path: Path) -> Path:
    target = dir_path / file_path.name
    if target.exists():
        target = dir_path / f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
    return Path(shutil.move(str(file_path), str(target)))


def create_bundle_dir(output_dir: Path) -> Path:
    base = time.strftime("%Y%m%d_%H%M%S")
    bundle_dir = output_dir / base
    if not bundle_dir.exists():
        bundle_dir.mkdir(parents=True, exist_ok=True)
        return bundle_dir
    counter = 1
    while True:
        candidate = output_dir / f"{base}_{counter}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        counter += 1


def process_once(input_dir: Path, output_dir: Path, failed_dir: Path,
                 model: str, response_format: str, prompt: str | None,
                 edit_model: str | None, edit_prompt: str | None,
                 min_age_seconds: float) -> int:
    client = OpenAI()
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        return 0
    processed_count = 0
    for path in files:
        if not is_file_ready(path, min_age_seconds):
            continue
        try:
            bundle_dir = create_bundle_dir(output_dir)
            text = transcribe_file(client, path, model, response_format, prompt)
            write_output(bundle_dir, f"{path.stem}.raw.txt", text)
            if edit_model and edit_prompt:
                edited_text = edit_transcript(client, text, edit_model, edit_prompt)
                write_output(bundle_dir, f"{path.stem}.edited.txt", edited_text)
            move_to(bundle_dir, path)
            processed_count += 1
            print(f"Transcribed: {path.name}")
        except Exception as exc:  # noqa: BLE001
            error_path = failed_dir / (path.stem + ".error.txt")
            error_path.write_text(str(exc), encoding="utf-8")
            move_to(failed_dir, path)
            print(f"Failed: {path.name} ({exc})", file=sys.stderr)
    return processed_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a folder and transcribe audio files with OpenAI.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent), help="Root folder")
    parser.add_argument("--once", action="store_true", help="Process current files and exit")
    parser.add_argument("--poll", type=float, default=float(os.getenv("POLL_SECONDS", "2")))
    parser.add_argument("--min-age", type=float, default=float(os.getenv("MIN_AGE_SECONDS", "1")))
    parser.add_argument("--model", default=os.getenv("MODEL", "gpt-4o-transcribe"))
    parser.add_argument("--response-format", default=os.getenv("RESPONSE_FORMAT", "text"))
    parser.add_argument("--prompt", default=os.getenv("PROMPT"))
    args = parser.parse_args()

    load_dotenv(Path(args.root).resolve() / ".env")

    root = Path(args.root).resolve()
    input_dir = root / "input"
    output_dir = root / "output"
    failed_dir = root / "failed"

    for d in (input_dir, output_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Set it in your environment and retry.", file=sys.stderr)
        return 2

    edit_model = os.getenv("EDIT_MODEL")
    edit_prompt = os.getenv("EDIT_PROMPT")
    if args.once:
        process_once(input_dir, output_dir, failed_dir,
                     args.model, args.response_format, args.prompt,
                     edit_model, edit_prompt,
                     args.min_age)
        return 0

    print(f"Watching: {input_dir}")
    while True:
        process_once(input_dir, output_dir, failed_dir,
                     args.model, args.response_format, args.prompt,
                     edit_model, edit_prompt,
                     args.min_age)
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())

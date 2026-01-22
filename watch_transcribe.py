#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
DEFAULT_DIARIZE_EDIT_PROMPT = (
    "Create a concise meeting summary with sections: Summary, Decisions, Action Items, "
    "Risks, and Open Questions. Action Items must be a bullet list with owner (if known) "
    "and due date (if mentioned). If something is unclear, mark it as [unclear]. "
    "Our company name is MODU Valves and we sell industrial valves - normalize references "
    "to the company and products accordingly."
)
FORMAT_TEMPLATES = {
    "meeting_summary": (
        "Create a concise meeting summary with sections: Summary, Decisions, Action Items, "
        "Risks, and Open Questions. Action Items must be a bullet list with owner (if known) "
        "and due date (if mentioned)."
    ),
    "action_items": (
        "Extract only Action Items as a bullet list. Each item should include owner (if known), "
        "due date (if mentioned), and the specific task."
    ),
    "internal_email": (
        "Write an internal email recap. Include a short subject line, brief context, key points, "
        "decisions, and action items."
    ),
    "sales_email": (
        "Write a sales follow-up email. Include a subject line, a brief recap, customer value, "
        "next steps, and a clear call to action."
    ),
    "linkedin_post": (
        "Write a LinkedIn post. Keep it professional and concise, include a short hook, 3-5 short "
        "paragraphs or bullets, and a closing line."
    ),
    "blog_post": (
        "Write a short blog post with a title, intro, 3-5 sections with headings, and a conclusion."
    ),
    "sop": (
        "Write an instruction/SOP/guideline. Use numbered steps, include prerequisites, and keep it "
        "actionable."
    ),
    "knowledge_piece": (
        "Write a product/industry/customer knowledge piece. Explain concepts clearly, include a few "
        "key takeaways, and keep it accurate."
    ),
}


def normalize_format_name(value: str) -> str:
    return re.sub(r"[-\\s]+", "_", value.strip().lower())


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def parse_formats(value: str | None) -> list[str]:
    if not value:
        return []
    raw = re.split(r"[,\n]", value)
    return [normalize_format_name(item) for item in raw if item.strip()]


def load_corrections_and_names(corrections_path: Path | None, names_path: Path | None) -> tuple[list[tuple[str, str]], list[str]]:
    corrections: list[tuple[str, str]] = []
    names: list[str] = []
    if corrections_path and corrections_path.exists():
        text = corrections_path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(text)
                items = payload.get("transcription_corrections", [])
                for item in items:
                    incorrect = str(item.get("incorrect", "")).strip()
                    correct = str(item.get("correct", "")).strip()
                    if incorrect and correct:
                        corrections.append((incorrect, correct))
            except json.JSONDecodeError:
                pass
        else:
            section: str | None = None
            for line in text.splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                upper = raw.upper()
                if upper.startswith("NAMES"):
                    section = "names"
                    continue
                if upper.startswith("CORRECTIONS"):
                    section = "corrections"
                    continue
                if raw.startswith("-"):
                    raw = raw[1:].strip()
                if section == "names":
                    names.append(raw)
                    continue
                if section == "corrections" and "->" in raw:
                    left, right = raw.split("->", 1)
                    left = left.strip()
                    right = right.strip()
                    if left and right:
                        corrections.append((left, right))
    if names_path and names_path.exists():
        for line in names_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("-"):
                raw = raw[1:].strip()
            names.append(raw)
    return corrections, names


def apply_corrections(text: str, corrections: list[tuple[str, str]]) -> str:
    result = text
    for incorrect, correct in corrections:
        variants = [part.strip() for part in incorrect.split("/") if part.strip()]
        if not variants:
            continue
        for variant in variants:
            pattern = re.compile(re.escape(variant), flags=re.IGNORECASE)
            result = pattern.sub(correct, result)
    return result


def build_prompt(format_name: str, base_prompt: str | None, tone: str, length: str, audience: str,
                 company_name: str, company_context: str,
                 names: list[str], corrections: list[tuple[str, str]]) -> str:
    if format_name == "custom":
        if not base_prompt:
            raise ValueError("EDIT_PROMPT is required for custom format.")
        template = base_prompt
    else:
        template = FORMAT_TEMPLATES.get(format_name)
        if not template:
            raise ValueError(f"Unknown format: {format_name}")
    lines = [
        template,
        f"Tone: {tone}. Length: {length}. Audience: {audience}.",
        f"Company context: {company_name}. {company_context}",
        "If something is unclear, mark it as [unclear].",
    ]
    if names:
        lines.append("People mentioned (spellings): " + "; ".join(names))
    if corrections:
        corrections_text = "; ".join([f"{left} -> {right}" for left, right in corrections])
        lines.append("Use corrected terms: " + corrections_text)
    return "\n".join(lines)


def is_file_ready(path: Path, min_age_seconds: float) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    if stat.st_size == 0:
        return False
    age = time.time() - stat.st_mtime
    return age >= min_age_seconds


def transcribe_file(client: OpenAI, path: Path, model: str, response_format: str,
                    prompt: str | None, chunking_strategy: str | None) -> object:
    with path.open("rb") as audio_file:
        kwargs = {
            "model": model,
            "file": audio_file,
            "response_format": response_format,
        }
        if prompt and "diarize" not in model:
            kwargs["prompt"] = prompt
        if chunking_strategy and "diarize" in model:
            kwargs["chunking_strategy"] = chunking_strategy
        transcription = client.audio.transcriptions.create(**kwargs)
    return transcription


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


def write_json(output_dir: Path, name: str, data: object) -> Path:
    output_path = output_dir / name
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path


def diarized_segments_to_text(segments: list[object]) -> str:
    lines: list[str] = []
    for segment in segments:
        speaker = getattr(segment, "speaker", None) if not isinstance(segment, dict) else segment.get("speaker")
        start = getattr(segment, "start", None) if not isinstance(segment, dict) else segment.get("start")
        end = getattr(segment, "end", None) if not isinstance(segment, dict) else segment.get("end")
        text = getattr(segment, "text", None) if not isinstance(segment, dict) else segment.get("text")
        speaker_label = speaker or "speaker"
        start_str = f"{float(start):.2f}" if isinstance(start, (int, float)) else "?"
        end_str = f"{float(end):.2f}" if isinstance(end, (int, float)) else "?"
        lines.append(f"{speaker_label} [{start_str}-{end_str}]: {text}")
    return "\n".join(lines)


def transcription_to_dict(transcription: object) -> dict:
    if hasattr(transcription, "model_dump"):
        return transcription.model_dump()
    if isinstance(transcription, dict):
        return transcription
    return {"text": str(transcription)}


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
                 chunking_strategy: str | None,
                 edit_model: str | None, edit_prompt: str | None,
                 edit_formats: list[str],
                 edit_tone: str, edit_length: str, edit_audience: str,
                 company_name: str, company_context: str,
                 corrections: list[tuple[str, str]], names: list[str],
                 use_corrections: bool,
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
            transcription = transcribe_file(client, path, model, response_format, prompt, chunking_strategy)
            transcription_dict = transcription_to_dict(transcription)
            text = transcription_dict.get("text") if isinstance(transcription_dict, dict) else None
            diarized_text = None
            if text:
                write_output(bundle_dir, f"{path.stem}.raw.txt", text)
            if response_format == "diarized_json":
                write_json(bundle_dir, f"{path.stem}.diarized.json", transcription_dict)
                segments = transcription_dict.get("segments", [])
                if segments:
                    diarized_text = diarized_segments_to_text(segments)
                    write_output(bundle_dir, f"{path.stem}.diarized.txt", diarized_text)
            if edit_model and (edit_prompt or edit_formats):
                base_text = text or diarized_text
                if not base_text:
                    raise ValueError("Edited transcript requested but transcription text is empty.")
                if use_corrections and corrections:
                    base_text = apply_corrections(base_text, corrections)
                formats = edit_formats[:]
                if not formats:
                    formats = ["custom"]
                single_format = len(formats) == 1
                for format_name in formats:
                    prompt_text = build_prompt(
                        format_name,
                        edit_prompt,
                        edit_tone,
                        edit_length,
                        edit_audience,
                        company_name,
                        company_context,
                        names,
                        corrections,
                    )
                    edited_text = edit_transcript(client, base_text, edit_model, prompt_text)
                    if single_format:
                        output_name = f"{path.stem}.edited.txt"
                    else:
                        output_name = f"{path.stem}.{format_name}.txt"
                    write_output(bundle_dir, output_name, edited_text)
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
    parser.add_argument("--chunking-strategy", default=os.getenv("CHUNKING_STRATEGY", "auto"))
    parser.add_argument("--format", default=os.getenv("FORMAT"))
    parser.add_argument("--formats", default=os.getenv("FORMATS"))
    parser.add_argument("--tone", default=os.getenv("TONE", "professional"))
    parser.add_argument("--length", default=os.getenv("LENGTH", "medium"))
    parser.add_argument("--audience", default=os.getenv("AUDIENCE", "internal"))
    parser.add_argument("--company-name", default=os.getenv("COMPANY_NAME", "MODU Valves"))
    parser.add_argument("--company-context", default=os.getenv("COMPANY_CONTEXT", "We sell industrial valves."))
    parser.add_argument("--corrections-file", default=os.getenv("CORRECTIONS_FILE", "corrections.txt"))
    parser.add_argument("--names-file", default=os.getenv("NAMES_FILE"))
    parser.add_argument("--use-corrections", action=argparse.BooleanOptionalAction,
                        default=parse_bool(os.getenv("USE_CORRECTIONS", "true"), True))
    args = parser.parse_args()

    load_dotenv(Path(args.root).resolve() / ".env")
    response_format = args.response_format
    if "diarize" in args.model and response_format == "text" and not os.getenv("RESPONSE_FORMAT"):
        response_format = "diarized_json"
    if "diarize" in args.model and args.prompt:
        print("PROMPT is ignored for diarization models.", file=sys.stderr)

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
    if "diarize" in args.model and edit_model and not edit_prompt:
        edit_prompt = DEFAULT_DIARIZE_EDIT_PROMPT
    chunking_strategy = args.chunking_strategy if "diarize" in args.model else None
    format_value = args.formats if args.formats else args.format
    edit_formats = parse_formats(format_value)
    corrections_path = Path(args.corrections_file) if args.corrections_file else None
    names_path = Path(args.names_file) if args.names_file else None
    corrections, names = load_corrections_and_names(corrections_path, names_path)
    if args.once:
        process_once(input_dir, output_dir, failed_dir,
                     args.model, response_format, args.prompt,
                     chunking_strategy,
                     edit_model, edit_prompt,
                     edit_formats,
                     args.tone, args.length, args.audience,
                     args.company_name, args.company_context,
                     corrections, names,
                     args.use_corrections,
                     args.min_age)
        return 0

    print(f"Watching: {input_dir}")
    while True:
        process_once(input_dir, output_dir, failed_dir,
                     args.model, response_format, args.prompt,
                     chunking_strategy,
                     edit_model, edit_prompt,
                     edit_formats,
                     args.tone, args.length, args.audience,
                     args.company_name, args.company_context,
                     corrections, names,
                     args.use_corrections,
                     args.min_age)
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())

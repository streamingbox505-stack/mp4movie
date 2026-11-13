#!/usr/bin/env python3
"""
Ngwele Pro Processor
====================
Pipeline:
  1. Fetch video from HF dataset  (jaynesmaxtv/movie-file)
  2. Remove / replace background  (rembg + custom BG image)
  3. Transcribe Swahili speech     (openai/whisper-large-v3)
  4. Burn stylish subtitles        (ffmpeg drawtext / ASS)
  5. Upload processed video back   (jaynesmaxtv/movie-file-pro)

Self-installing — no requirements.txt needed.
"""

import subprocess, sys

def _install():
    pkgs = [
        "rembg",
        "openai-whisper",
        "torch",
        "torchvision",
        "opencv-python-headless",
        "Pillow",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "gdown",
        "requests",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + pkgs,
        check=True
    )

_install()

import json, os, re, shutil, subprocess as sp, time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from rembg import new_session, remove
import whisper
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
SRC_REPO      = os.getenv("HF_SRC_REPO",  "jaynesmaxtv/movie-file")
DST_REPO      = os.getenv("HF_DST_REPO",  "jaynesmaxtv/movie-file-pro")
HF_TOKEN      = os.environ["HF_TOKEN"]
BATCH         = int(os.getenv("BATCH", "5"))
BG_IMAGE      = os.getenv("BG_IMAGE", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
START         = time.time()
DEADLINE      = START + int(os.getenv("DEADLINE_MIN", "300")) * 60
TMP           = Path("work_pro")
api           = HfApi(token=HF_TOKEN)

ASS_STYLE = (
    "Style: Default,Noto Sans,22,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
    "0,0,0,0,100,100,0,0,1,2.5,0.5,2,10,10,25,1"
)


def log(*a):
    print(*a, flush=True)


# ── 1. Fetch todo list ────────────────────────────────────────────────────────
def fetch_todo():
    src_files = set(api.list_repo_files(SRC_REPO, repo_type="dataset"))
    try:
        dst_files = set(api.list_repo_files(DST_REPO, repo_type="dataset"))
    except Exception:
        dst_files = set()
    return [
        f for f in src_files
        if f.startswith("videos/") and f.endswith(".mp4")
        and f not in dst_files
    ]


# ── 2. Background removal ─────────────────────────────────────────────────────
def load_background(path, w, h):
    if path and os.path.exists(path):
        return np.array(Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS))
    # Default: cinematic dark-blue gradient
    arr = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        t = y / h
        arr[y] = [int(5+10*t), int(5+5*t), int(30+60*t)]
    return arr


def remove_bg_video(input_path, output_path, bg_image_path=""):
    log("  → removing background ...")
    cap   = cv2.VideoCapture(input_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg    = load_background(bg_image_path, w, h).astype(np.float32)

    session  = new_session("u2net_human_seg")
    tmp_vid  = output_path.replace(".mp4", "_novid.mp4")
    writer   = cv2.VideoWriter(tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if n % 50 == 0:
            log(f"     frame {n}/{total}")
        pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rgba  = np.array(remove(pil, session=session)).astype(np.float32)
        alpha = rgba[:, :, 3:4] / 255.0
        comp  = (rgba[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
        writer.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()

    sp.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", tmp_vid, "-i", input_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path
    ], check=True)
    os.remove(tmp_vid)
    log("  ✓ background removed")


# ── 3. Swahili transcription ──────────────────────────────────────────────────
def transcribe_swahili(video_path):
    log("  → transcribing Swahili audio ...")
    wav = video_path.replace(".mp4", "_audio.wav")
    sp.run(["ffmpeg", "-y", "-v", "error", "-i", video_path, "-ac", "1", "-ar", "16000", wav], check=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = whisper.load_model(WHISPER_MODEL, device=device)
    result = model.transcribe(wav, language="sw", task="transcribe", verbose=False)
    os.remove(wav)
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result["segments"] if s["text"].strip()]
    log(f"  ✓ {len(segs)} segments transcribed")
    return segs


# ── 4. Build ASS subtitle file ────────────────────────────────────────────────
def _ts(t):
    h, m, s = int(t // 3600), int((t % 3600) // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(segments, out_path, vw=1280, vh=720):
    lines = [
        "[Script Info]", "ScriptType: v4.00+",
        f"PlayResX: {vw}", f"PlayResY: {vh}", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        ASS_STYLE, "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for seg in segments:
        words, cur, wrapped = seg["text"].split(), [], []
        for w in words:
            cur.append(w)
            if len(" ".join(cur)) > 45:
                wrapped.append(" ".join(cur[:-1])); cur = [w]
        if cur:
            wrapped.append(" ".join(cur))
        lines.append(
            f"Dialogue: 0,{_ts(seg['start'])},{_ts(seg['end'])},"
            f"Default,,0,0,0,,{chr(92)+'N'.join(wrapped)}"
        )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    log(f"  ✓ subtitles written")


# ── 5. Burn subtitles ─────────────────────────────────────────────────────────
def burn_subtitles(video_in, ass_path, video_out):
    log("  → burning subtitles ...")
    sp.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", video_in, "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", video_out
    ], check=True)
    log("  ✓ subtitles burned")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def process_one(hf_path):
    log(f"\n▶ Processing: {hf_path}")
    TMP.mkdir(exist_ok=True)

    local_src = str(TMP / "source.mp4")
    hf_hub_download(SRC_REPO, hf_path, repo_type="dataset", local_dir=str(TMP), token=HF_TOKEN)
    saved = TMP / hf_path
    if saved.exists():
        shutil.move(str(saved), local_src)

    bg_removed = str(TMP / "bg_removed.mp4")
    final_out  = str(TMP / "final.mp4")
    ass_file   = str(TMP / "subs.ass")

    remove_bg_video(local_src, bg_removed, BG_IMAGE)
    segs = transcribe_swahili(local_src)

    cap = cv2.VideoCapture(bg_removed)
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    build_ass(segs, ass_file, vw, vh)
    burn_subtitles(bg_removed, ass_file, final_out)

    transcript_path = str(TMP / "transcript.json")
    json.dump(segs, open(transcript_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    dst_vid  = hf_path.replace("videos/ngwele/", "videos/ngwele-pro/")
    dst_json = dst_vid.replace(".mp4", "_transcript.json")

    api.upload_file(path_or_fileobj=final_out,        path_in_repo=dst_vid,
                    repo_id=DST_REPO, repo_type="dataset", commit_message=f"pro: {dst_vid}")
    api.upload_file(path_or_fileobj=transcript_path,  path_in_repo=dst_json,
                    repo_id=DST_REPO, repo_type="dataset", commit_message=f"transcript: {dst_json}")

    log(f"  ✓ uploaded → {dst_vid}")
    shutil.rmtree(str(TMP), ignore_errors=True)


def main():
    todo = fetch_todo()
    log(f"{len(todo)} videos pending, batch={BATCH}")
    done = 0
    for path in todo:
        if done >= BATCH or time.time() > DEADLINE:
            break
        try:
            process_one(path)
            done += 1
        except Exception as e:
            log(f"  ✗ FAILED {path}: {str(e)[:300]}")
    log(f"\n✅ Done: {done} processed in {(time.time()-START)/60:.1f} min")


if __name__ == "__main__":
    main()

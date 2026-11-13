#!/usr/bin/env python3
"""
Ngwele Pro — Hugging Face Space
================================
Upload a video → remove background → Swahili subtitles → download pro version.

Self-installing — no requirements.txt needed.
"""

import subprocess, sys

def _install():
    pkgs = [
        "gradio>=4.44.0",
        "rembg",
        "openai-whisper",
        "torch",
        "torchvision",
        "opencv-python-headless",
        "Pillow",
        "numpy",
        "soundfile",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + pkgs,
        check=True
    )

_install()

import os, json, re, subprocess as sp, tempfile, shutil
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import cv2
import whisper
from PIL import Image
from rembg import new_session, remove as rembg_remove

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
ASS_STYLE = (
    "Style: Default,Noto Sans,24,&H00FFFFFF,&H000000FF,&H00000000,&HCC000000,"
    "0,0,0,0,100,100,0,0,1,2.8,0.6,2,10,10,30,1"
)

_whisper_cache = {}
_rembg_session = None


def get_whisper():
    if "model" not in _whisper_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _whisper_cache["model"] = whisper.load_model(WHISPER_MODEL, device=device)
    return _whisper_cache["model"]


def get_rembg():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("u2net_human_seg")
    return _rembg_session


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_gradient_bg(w, h, top="#0a0a2e", bottom="#1a1a4e"):
    tr, tg, tb = int(top[1:3],16), int(top[3:5],16), int(top[5:7],16)
    br, bg_, bb = int(bottom[1:3],16), int(bottom[3:5],16), int(bottom[5:7],16)
    arr = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        t = y / h
        arr[y] = [int(tr+(br-tr)*t), int(tg+(bg_-tg)*t), int(tb+(bb-tb)*t)]
    return arr


def _ts(t):
    h, m, s = int(t//3600), int((t%3600)//60), t%60
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
            if len(" ".join(cur)) > 42:
                wrapped.append(" ".join(cur[:-1])); cur = [w]
        if cur:
            wrapped.append(" ".join(cur))
        lines.append(
            f"Dialogue: 0,{_ts(seg['start'])},{_ts(seg['end'])},"
            f"Default,,0,0,0,,{chr(92)+'N'.join(wrapped)}"
        )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ── Core pipeline ─────────────────────────────────────────────────────────────
def process(video_path, bg_choice, custom_bg, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Pakia video kwanza.")

    work = Path(tempfile.mkdtemp())
    try:
        progress(0.05, desc="Inachunguza video...")
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress(0.10, desc="Inaandaa background...")
        if bg_choice == "Gradient ya Sinema" or not custom_bg:
            bg_arr = make_gradient_bg(w, h).astype(np.float32)
        else:
            bg_arr = np.array(Image.open(custom_bg).convert("RGB").resize((w, h), Image.LANCZOS)).astype(np.float32)

        progress(0.15, desc="Inaondoa background (inaweza chukua muda)...")
        session = get_rembg()
        tmp_vid = str(work / "nobg.mp4")
        writer  = cv2.VideoWriter(tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        n = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rgba  = np.array(rembg_remove(pil, session=session)).astype(np.float32)
            alpha = rgba[:, :, 3:4] / 255.0
            comp  = (rgba[:, :, :3] * alpha + bg_arr * (1 - alpha)).astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
            if n % max(1, total // 20) == 0:
                progress(0.15 + 0.45 * (n / total), desc=f"Frame {n}/{total}...")

        cap.release(); writer.release()

        merged = str(work / "merged.mp4")
        sp.run([
            "ffmpeg", "-y", "-v", "error",
            "-i", tmp_vid, "-i", video_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", merged
        ], check=True)

        progress(0.65, desc="Inasikia Kiswahili (Whisper)...")
        wav = str(work / "audio.wav")
        sp.run(["ffmpeg", "-y", "-v", "error", "-i", video_path, "-ac", "1", "-ar", "16000", wav], check=True)
        model  = get_whisper()
        result = model.transcribe(wav, language="sw", task="transcribe", verbose=False)
        segs   = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                  for s in result["segments"] if s["text"].strip()]
        os.remove(wav)

        progress(0.80, desc="Inaweka maandishi...")
        ass = str(work / "subs.ass")
        build_ass(segs, ass, w, h)

        final = str(work / "final.mp4")
        sp.run([
            "ffmpeg", "-y", "-v", "error",
            "-i", merged, "-vf", f"ass={ass}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart", final
        ], check=True)

        tj = str(work / "transcript.json")
        json.dump(segs, open(tj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        progress(1.0, desc="Imekamilika! ✅")

        out_vid = "/tmp/ngwele_final.mp4"
        out_jsn = "/tmp/ngwele_transcript.json"
        shutil.copy(final, out_vid)
        shutil.copy(tj, out_jsn)
        return out_vid, out_jsn, f"✅ Imefanikiwa — maneno {len(segs)} yameandikwa."

    except Exception as e:
        raise gr.Error(f"Hitilafu: {str(e)[:400]}")
    finally:
        shutil.rmtree(str(work), ignore_errors=True)


# ── UI ────────────────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'Inter', sans-serif; }
#header { background: linear-gradient(135deg,#0d0d2b,#1a1a4e);
          padding: 2rem; border-radius: 12px; margin-bottom: 1rem; }
#header h1 { color: #fff; font-size: 2rem; margin:0; letter-spacing:-0.5px; }
#header p  { color: #aab; margin:0.4rem 0 0; font-size:0.95rem; }
"""

with gr.Blocks(css=CSS, title="Ngwele Pro") as demo:
    gr.HTML("""
    <div id="header">
      <h1>🎬 Ngwele Pro</h1>
      <p>Ondoa background · Maandishi ya Kiswahili kiotomatiki · Video ya ubora wa juu</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp_video = gr.Video(label="📤 Pakia Video", height=300)
            bg_choice = gr.Radio(
                ["Gradient ya Sinema", "Picha Yako Mwenyewe"],
                value="Gradient ya Sinema",
                label="🖼️ Chagua Background"
            )
            custom_bg = gr.Image(
                label="Pakia Picha ya Background (hiari)",
                type="filepath",
                visible=False
            )
            bg_choice.change(
                lambda c: gr.update(visible=c == "Picha Yako Mwenyewe"),
                inputs=bg_choice, outputs=custom_bg
            )
            btn = gr.Button("🚀 Fanya Pro!", variant="primary", size="lg")

        with gr.Column(scale=1):
            out_video   = gr.Video(label="✅ Video ya Pro", height=300)
            out_json    = gr.File(label="📄 Transcript (JSON)")
            status_text = gr.Markdown("")

    btn.click(
        process,
        inputs=[inp_video, bg_choice, custom_bg],
        outputs=[out_video, out_json, status_text]
    )

    gr.Markdown("""
    ---
    **Vidokezo:**
    - Video ndefu zinachukua muda zaidi (rembg inafanya kila frame)
    - Whisper `medium` inafanya kazi vizuri na Kiswahili
    - Transcript JSON inaweza kutumika tena bila kufanya upya
    """)


if __name__ == "__main__":
    demo.launch()

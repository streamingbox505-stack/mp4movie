#!/usr/bin/env python3
"""dub_movie.py — Dubbing ya filamu kwa Kiswahili, bure kabisa (inakimbia kwenye GitHub Actions, CPU tu).

Mnyororo: ffmpeg (sauti) -> faster-whisper (maandishi + muda) -> NLLB-200 (tafsiri kwenda Kiswahili)
          -> MMS-TTS swh (sauti ya Kiswahili) -> ffmpeg (kuchanganya na muziki/sauti ya asili) -> mp4 mpya + .vtt -> Hugging Face

LEseni: NLLB na MMS-TTS ni CC-BY-NC 4.0 — matumizi yasiyo ya kibiashara tu, na lazima utoe sifa kwa Meta AI.
        Ukianza kupata mapato, badilisha sauti/tafsiri kwa zenye leseni ya kibiashara.

Env: SRC_URL (au njia ya faili), NAME, SRC_LANG (hiari), BG_VOLUME (0.25), MAX_SPEED (1.5), WHISPER_MODEL (small),
     KEEP_ORIGINAL (0/1), HF_TOKEN, DUB_REPO (jaynesmaxtv/movie-file), DUB_PREFIX (dubbed/), SKIP_UPLOAD (0/1)
"""
import json, os, re, shutil, subprocess, sys, time, urllib.request, wave
import numpy as np

E = os.environ.get
SRC, NAME = E("SRC_URL", "").strip(), re.sub(r"[^A-Za-z0-9_-]+", "-", E("NAME", "movie")).strip("-") or "movie"
SRC_LANG = E("SRC_LANG", "").strip() or None
BG, MAX_SPEED, WHISPER = float(E("BG_VOLUME", "0.25")), float(E("MAX_SPEED", "1.5")), E("WHISPER_MODEL", "small")
KEEP_ORIG = E("KEEP_ORIGINAL", "0") == "1"
TOKEN, DUB_REPO, PREFIX = E("HF_TOKEN", "").strip(), E("DUB_REPO", "jaynesmaxtv/movie-file"), E("DUB_PREFIX", "dubbed/")
NLLB, TTS_MODEL, WORK, SR = "facebook/nllb-200-distilled-600M", "facebook/mms-tts-swh", E("WORK_DIR", "work"), 16000
NLLB_CODES = {"en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn", "pt": "por_Latn", "de": "deu_Latn", "it": "ita_Latn", "ru": "rus_Cyrl",
              "ar": "arb_Arab", "hi": "hin_Deva", "ko": "kor_Hang", "ja": "jpn_Jpan", "zh": "zho_Hans", "tr": "tur_Latn", "nl": "nld_Latn",
              "pl": "pol_Latn", "id": "ind_Latn", "th": "tha_Thai", "vi": "vie_Latn", "sv": "swe_Latn", "uk": "ukr_Cyrl", "fa": "pes_Arab"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} imeshindwa: {(r.stderr or b'').decode(errors='replace')[-600:]}")
    return r


def probe(path, entry="format=duration"):
    return run(["ffprobe", "-v", "error", "-show_entries", entry, "-of", "csv=p=0", path]).stdout.decode().strip()


def download(src, dest):
    if os.path.exists(src):
        return src
    req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
    if TOKEN and "huggingface.co" in src:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    return dest


def load_wav(path):
    """Soma wav ya 16kHz mono kama safu ya float32 (Whisper inaipokea moja kwa moja, bila kutegemea maktaba ya av)."""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, "asr.wav si 16kHz mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


# ---------- hatua za AI (zinapakia models kwa uvivu) ----------
def transcribe(wav):
    from faster_whisper import WhisperModel
    model = WhisperModel(WHISPER, device="cpu", compute_type="int8")
    segs, info = model.transcribe(load_wav(wav), language=SRC_LANG, vad_filter=True, beam_size=1, condition_on_previous_text=False)
    out, last = [], 0
    for s in segs:
        t = s.text.strip()
        if t and s.no_speech_prob < 0.6 and s.end > s.start:
            out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": t})
        if s.end - last >= 600:
            last = s.end; log(f"  … imefika dakika {s.end / 60:.0f}")
    return out, info.language


def translate(texts, lang):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    code = NLLB_CODES.get(lang, "eng_Latn")
    tok, mdl = AutoTokenizer.from_pretrained(NLLB, src_lang=code), AutoModelForSeq2SeqLM.from_pretrained(NLLB).eval()
    bos, out = tok.convert_tokens_to_ids("swh_Latn"), []
    for i in range(0, len(texts), 8):
        b = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=200)
        with torch.inference_mode():
            g = mdl.generate(**b, forced_bos_token_id=bos, num_beams=2, max_new_tokens=220)
        out += tok.batch_decode(g, skip_special_tokens=True)
        if (i // 8) % 25 == 0:
            log(f"  … tafsiri {min(i + 8, len(texts))}/{len(texts)}")
    return out


class Voice:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, VitsModel
        self.torch, self.tok, self.m = torch, AutoTokenizer.from_pretrained(TTS_MODEL), VitsModel.from_pretrained(TTS_MODEL).eval()
        self.sr = int(self.m.config.sampling_rate)

    def __call__(self, text):
        parts = [p.strip() for p in re.split(r"(?<=[.!?;:])\s+", text) if p.strip()]
        chunks = []
        for p in parts:
            while len(p) > 200:                      # vipande virefu: kata kwenye nafasi
                cut = p.rfind(" ", 0, 200)
                cut = cut if cut > 0 else 200
                chunks.append(p[:cut]); p = p[cut:].strip()
            chunks.append(p)
        wavs = []
        for c in chunks:
            try:
                self.torch.manual_seed(0)
                with self.torch.inference_mode():
                    w = self.m(**self.tok(c, return_tensors="pt")).waveform[0].cpu().numpy().astype(np.float32)
                wavs += [w, np.zeros(int(0.12 * self.sr), np.float32)]
            except Exception:                        # kipande kisicho na herufi zinazojulikana
                continue
        if not wavs:
            return None
        w = np.concatenate(wavs[:-1])
        return w if self.sr == SR else np.interp(np.linspace(0, len(w) - 1, int(len(w) * SR / self.sr)), np.arange(len(w)), w).astype(np.float32)


# ---------- sauti: kupanga kwenye ratiba, kuchanganya ----------
def stretch(a, ratio):
    """Kuharakisha bila kubadilisha sauti (atempo)."""
    r = run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0", "-filter:a", f"atempo={ratio:.3f}",
             "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"], input=a.astype(np.float32).tobytes())
    return np.frombuffer(r.stdout, np.float32).copy()


def build_track(items, total):
    """items = [(start, end, audio)] -> wimbi moja lenye ratiba. Sentensi ndefu huharakishwa (hadi MAX_SPEED), zisizoingia husogeza kinachofuata."""
    track, cursor = np.zeros(int(total * SR) + SR, np.float32), 0.0
    for i, (s, e, a) in enumerate(items):
        nxt = items[i + 1][0] if i + 1 < len(items) else total
        avail = max(e - s, nxt - s - 0.15, 0.3)
        if len(a) / SR > avail:
            a = stretch(a, min(len(a) / SR / avail, MAX_SPEED))
        pos = int(max(s, cursor) * SR)
        if pos + len(a) > len(track):
            track = np.pad(track, (0, pos + len(a) - len(track) + SR))
        track[pos:pos + len(a)] += a
        cursor = (pos + len(a)) / SR + 0.05
    return track


def write_wav(x, path):
    pk = float(np.max(np.abs(x))) or 1.0
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x / pk * 0.95, -1, 1) * 32767).astype(np.int16).tobytes())


def mix(src, voice_wav, out):
    has_audio = bool(probe(src, "stream=codec_type").count("audio"))
    vf = "aresample=48000,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,aformat=channel_layouts=stereo"
    if has_audio:   # sauti ya asili hupungua yenyewe pale Kiswahili kinaposemwa (ducking), kisha inachanganywa
        graph = (f"[1:a]{vf},asplit=2[v1][v2];[0:a:0]aresample=48000,aformat=channel_layouts=stereo[o];"
                 f"[o][v1]sidechaincompress=threshold=0.02:ratio=10:attack=20:release=400[d];[d]volume={BG}[bg];"
                 f"[bg][v2]amix=inputs=2:duration=first:normalize=0[a]")
    else:
        graph = f"[1:a]{vf}[a]"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-i", voice_wav, "-filter_complex", graph, "-map", "0:v:0", "-map", "[a]"]
    if KEEP_ORIG and has_audio:
        cmd += ["-map", "0:a:0", "-metadata:s:a:1", "language=eng", "-metadata:s:a:1", "title=Original"]
    cmd += ["-metadata:s:a:0", "language=swa", "-metadata:s:a:0", "title=Kiswahili", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", out]
    run(cmd)


def vtt(items, path):
    ts = lambda t: f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s, e, text in items:
            f.write(f"{ts(s)} --> {ts(e)}\n{text}\n\n")


def main():
    if not SRC:
        sys.exit("SRC_URL haipo")
    os.makedirs(WORK, exist_ok=True)
    src = download(SRC, f"{WORK}/source.mp4")
    total = float(probe(src))
    log(f"▶️ {NAME}: {total / 60:.1f} dk")
    asr = f"{WORK}/asr.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(SR), asr])
    log("🎙️ Kutambua mazungumzo (Whisper)…")
    segs, lang = transcribe(asr)
    json.dump({"lang": lang, "segments": segs}, open(f"{WORK}/segments.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"   vipande {len(segs)}, lugha: {lang}")
    if not segs:
        sys.exit("Hakuna mazungumzo yaliyotambuliwa")
    if lang == "sw":
        sw = [s["text"] for s in segs]
    else:
        log("🌍 Kutafsiri kwenda Kiswahili (NLLB)…")
        sw = translate([s["text"] for s in segs], lang)
    json.dump([{**s, "sw": t} for s, t in zip(segs, sw)], open(f"{WORK}/sw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("🔊 Kutengeneza sauti ya Kiswahili (MMS-TTS)…")
    voice, items = Voice(), []
    for i, (s, t) in enumerate(zip(segs, sw)):
        a = voice(t)
        if a is not None and len(a) > 0:
            items.append((s["start"], s["end"], a))
        if i % 100 == 0:
            log(f"  … sauti {i}/{len(segs)}")
    voice_wav, out = f"{WORK}/voice.wav", f"{WORK}/{NAME}-sw.mp4"
    write_wav(build_track(items, total), voice_wav)
    log("🎚️ Kuchanganya na video…")
    mix(src, voice_wav, out)
    vtt([(s["start"], s["end"], t) for s, t in zip(segs, sw)], f"{WORK}/{NAME}-sw.vtt")
    log(f"✅ Tayari: {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    if E("SKIP_UPLOAD") == "1" or not TOKEN:
        log("ℹ️ Kupakia kumerukwa (hakuna HF_TOKEN au SKIP_UPLOAD=1)"); return
    from huggingface_hub import HfApi
    api = HfApi(token=TOKEN)
    api.create_repo(DUB_REPO, repo_type="dataset", exist_ok=True)
    for f in (f"{NAME}-sw.mp4", f"{NAME}-sw.vtt"):
        api.upload_file(path_or_fileobj=f"{WORK}/{f}", path_in_repo=f"{PREFIX}{f}", repo_id=DUB_REPO, repo_type="dataset")
    log(f"📦 https://huggingface.co/datasets/{DUB_REPO}/resolve/main/{PREFIX}{NAME}-sw.mp4")


if __name__ == "__main__":
    main()

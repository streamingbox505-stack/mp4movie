#!/usr/bin/env python3
"""dub_movie.py — Dubbing ya filamu kwa Kiswahili, bure kabisa (inakimbia kwenye GitHub Actions, CPU tu).

Mnyororo: ffmpeg (sauti) -> faster-whisper (maandishi + muda) -> NLLB-200 (tafsiri kwenda Kiswahili)
          -> MMS-TTS swh (sauti ya Kiswahili) -> ffmpeg (kuchanganya na muziki/sauti ya asili) -> mp4 mpya + .vtt -> Hugging Face

LEseni: NLLB na MMS-TTS ni CC-BY-NC 4.0 — matumizi yasiyo ya kibiashara tu, na lazima utoe sifa kwa Meta AI.
        Ukianza kupata mapato, badilisha sauti/tafsiri kwa zenye leseni ya kibiashara.

Dubbing ya kweli: Demucs huondoa sauti ya waigizaji wa asili (inabaki muziki + kelele), sauti ya Kiswahili ya kike/kiume
huchaguliwa kwa kila sentensi kutokana na sauti ya asili (pitch), na hisia za msingi (sauti kubwa/ndogo, juu/chini) huhamishwa.

Env: SRC_URL (au njia ya faili), NAME, SRC_LANG (hiari), WHISPER_MODEL (small), MAX_SPEED (1.5), KEEP_ORIGINAL (0/1)
     SEPARATE (1)       kuondoa sauti ya asili kwa Demucs (nzito: saa kadhaa kwa filamu ndefu); 0 = njia ya zamani (voice-over)
     BG_SEP (0.9)       sauti ya muziki/kelele baada ya kutenganisha;  BG_VOLUME (0.25) = njia ya zamani
     VOICE_F, VOICE_M   Hugging Face repo za sauti ya kike na kiume (muundo wa MMS/VITS); chaguo-msingi: facebook/mms-tts-swh
     PITCH_F (1.5), PITCH_M (-2.5)  semitones za msingi ili sauti zitofautiane hata kama modeli ni moja
     GENDER_SPLIT (165) Hz: pitch juu ya hii = kike;  EXPRESSIVE (1) = hamisha sauti kubwa/ndogo na pitch ya asili;  VOICE_GAIN (1.0)
     HF_TOKEN, DUB_REPO (jaynesmaxtv/movie-file), DUB_PREFIX (dubbed/), SKIP_UPLOAD (0/1)
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
SEPARATE, BG_SEP, VOICE_GAIN = E("SEPARATE", "1") == "1", float(E("BG_SEP", "0.9")), float(E("VOICE_GAIN", "1.0"))
VOICE_F, VOICE_M = E("VOICE_F", "").strip() or TTS_MODEL, E("VOICE_M", "").strip() or TTS_MODEL
PITCH = {"f": float(E("PITCH_F", "1.5")), "m": float(E("PITCH_M", "-2.5"))}
GENDER_SPLIT, EXPRESSIVE = float(E("GENDER_SPLIT", "165")), E("EXPRESSIVE", "1") == "1"
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
    def __init__(self, repo=TTS_MODEL):
        import torch
        from transformers import AutoTokenizer, VitsModel
        self.torch, self.tok, self.m = torch, AutoTokenizer.from_pretrained(repo), VitsModel.from_pretrained(repo).eval()
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
def fx(a, speed=1.0, semitones=0.0):
    """Kuharakisha (speed>1) na/au kubadilisha pitch (semitones) kwa ffmpeg; muda unabaki kulingana na 'speed' tu."""
    r = 2 ** (semitones / 12)
    if abs(r - 1) < 0.01 and abs(speed - 1) < 0.01:
        return a
    chain = (f"asetrate={SR * r:.1f},aresample={SR}," if abs(r - 1) >= 0.01 else "") + f"atempo={max(0.5, speed / r):.4f}"
    out = run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0", "-filter:a", chain,
               "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"], input=a.astype(np.float32).tobytes())
    return np.frombuffer(out.stdout, np.float32).copy()


def stretch(a, ratio):
    return fx(a, ratio, 0.0)


# ---------- kutenganisha sauti (Demucs) ----------
def separate(src, total):
    """Tenganisha sauti ya mazungumzo na muziki/kelele. Inarudisha (voc_raw, nov_raw): s16le, 44.1kHz, stereo. Inafanya kazi kwa vipande vya dakika 5."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    torch.set_num_threads(os.cpu_count() or 2)
    model = get_model("htdemucs").eval()
    sr, vi = int(model.samplerate), model.sources.index("vocals")
    CH, OV, t0, t = 300.0, 1.5, time.time(), 0.0
    vp, np_ = f"{WORK}/voc.raw", f"{WORK}/nov.raw"
    with open(vp, "wb") as fv, open(np_, "wb") as fn:
        while t < total:
            a, b = max(0.0, t - OV), min(total, t + CH + OV)
            raw = run(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", src, "-vn", "-f", "f32le", "-ac", "2", "-ar", str(sr), "pipe:1"]).stdout
            x = torch.from_numpy(np.frombuffer(raw, np.float32).copy()).view(-1, 2).t().contiguous()
            ref = x.mean(0)
            with torch.inference_mode():
                y = apply_model(model, ((x - ref.mean()) / (ref.std() + 1e-8))[None], split=True, overlap=0.1, device="cpu", progress=False)[0]
            v = y[vi] * (ref.std() + 1e-8) + ref.mean()
            lo = int((t - a) * sr)
            hi = min(v.shape[1], lo + int(min(CH, total - t) * sr))
            for sig, f in ((v, fv), (x - v, fn)):
                f.write((sig[:, lo:hi].t().contiguous().clamp(-1, 1) * 32767).short().numpy().tobytes())
            t += CH
            log(f"  … kutenganisha sauti: dakika {min(t, total) / 60:.0f}/{total / 60:.0f} (imepita {(time.time() - t0) / 60:.0f} dk)")
    return vp, np_


# ---------- sauti ya asili: pitch (kike/kiume) na kiwango cha sauti ----------
def f0_median(x):
    """Pitch ya wastani (Hz) ya sehemu yenye usemi, au None kama haitoshi."""
    fr, hop, lo, hi = 1024, 512, int(SR / 400), int(SR / 70)
    n = (len(x) - fr) // hop
    if n < 5:
        return None
    X = np.stack([x[i * hop:i * hop + fr] for i in range(n)])
    X = X - X.mean(1, keepdims=True)
    e = np.sqrt((X ** 2).mean(1))
    X, win, f = X[e > max(e.max() * 0.25, 1e-4)], np.hanning(fr), []
    for row in X:
        ac = np.fft.irfft(np.abs(np.fft.rfft(row * win, 2 * fr)) ** 2)[:fr]
        ac = ac / (ac[0] + 1e-9)
        seg = ac[lo:hi]
        k = lo + int(np.where(seg >= 0.9 * seg.max())[0][0])
        if ac[k] > 0.45:
            f.append(SR / k)
    return float(np.median(f)) if len(f) >= 4 else None


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
    pk = float(np.percentile(np.abs(x), 99.9)) or 1.0       # si 'peak' kamili, ili sauti moja kubwa isinyamazishe zote
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


def mix_sep(src, nov_raw, voice_wav, out):
    """Muziki/kelele (bila mazungumzo ya asili) + sauti ya Kiswahili, juu ya video ya asili (video haibadilishwi)."""
    graph = (f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={BG_SEP}[bg];"
             f"[2:a]aresample=48000,aformat=channel_layouts=stereo,volume={VOICE_GAIN}[v];"
             f"[bg][v]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.97[a]")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", nov_raw, "-i", voice_wav,
           "-filter_complex", graph, "-map", "0:v:0", "-map", "[a]"]
    if KEEP_ORIG and probe(src, "stream=codec_type").count("audio"):
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
    log(f"▶️ {NAME}: {total / 60:.1f} dk | kutenganisha sauti: {'ndiyo' if SEPARATE else 'hapana (voice-over)'}")
    asr, nov = f"{WORK}/asr.wav", None
    if SEPARATE:
        try:
            log("🎚️ Kutenganisha mazungumzo na muziki (Demucs)…")
            voc, nov = separate(src, total)
            run(["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", voc, "-ac", "1", "-ar", str(SR), asr])
            os.remove(voc)
        except Exception as e:                    # Demucs ikishindwa, endelea na njia ya zamani badala ya kupoteza kazi yote
            log(f"⚠️ Kutenganisha sauti kumeshindwa ({str(e)[:300]}). Naendelea na voice-over (sauti ya asili inapunguzwa).")
            nov = None
    if nov is None:
        run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(SR), asr])
    log("🎙️ Kutambua mazungumzo (Whisper)…")
    segs, lang = transcribe(asr)
    json.dump({"lang": lang, "segments": segs}, open(f"{WORK}/segments.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"   vipande {len(segs)}, lugha: {lang}")
    if not segs:
        sys.exit("Hakuna mazungumzo yaliyotambuliwa")
    cover = sum(s["end"] - s["start"] for s in segs) / total
    log(f"   mazungumzo yaliyotambuliwa: {cover * 100:.0f}% ya filamu")
    if lang == "sw" and E("FORCE_SW") != "1":
        sys.exit("⛔ Filamu hii inaonekana tayari ni ya Kiswahili (DJ). Whisper hasomi Kiswahili vizuri, kwa hiyo kuirudia kungeharibu sauti. "
                 "Tumia filamu ya asili (Kiingereza n.k.), au weka FORCE_SW=1 kama una uhakika.")
    if lang == "sw":
        sw = [s["text"] for s in segs]
    else:
        log("🌍 Kutafsiri kwenda Kiswahili (NLLB)…")
        sw = translate([s["text"] for s in segs], lang)

    # --- uchambuzi wa sauti ya asili: kike/kiume + kiwango cha sauti + pitch ---
    audio = load_wav(asr)
    info = []
    for s in segs:
        x = audio[int(s["start"] * SR):int(min(s["end"], s["start"] + 8) * SR)]
        info.append({"f0": f0_median(x) if len(x) > SR // 2 else None, "rms": float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0})
    del audio
    prev = "m"
    for k in info:
        k["g"] = ("f" if k["f0"] >= GENDER_SPLIT else "m") if k["f0"] else prev
        prev = k["g"]
    med_f0 = {g: float(np.median([k["f0"] for k in info if k["g"] == g and k["f0"]] or [150.0])) for g in "fm"}
    rms_all = [k["rms"] for k in info if k["rms"] > 0]
    med_rms = float(np.median(rms_all)) if rms_all else 1.0
    nf = sum(1 for k in info if k["g"] == "f")
    log(f"   sauti: kike {nf}, kiume {len(info) - nf} | pitch ya wastani: kike {med_f0['f']:.0f} Hz, kiume {med_f0['m']:.0f} Hz")
    json.dump([{**s, "sw": t, **k} for s, t, k in zip(segs, sw, info)], open(f"{WORK}/sw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    log("🔊 Kutengeneza sauti za Kiswahili (kike na kiume)…")
    voices = {"f": Voice(VOICE_F)}
    voices["m"] = voices["f"] if VOICE_M == VOICE_F else Voice(VOICE_M)
    items = []
    for i, (s, t, k) in enumerate(zip(segs, sw, info)):
        a = voices[k["g"]](t)
        if a is not None and len(a) > 0:
            a = a / (float(np.sqrt(np.mean(a ** 2))) + 1e-9) * 0.1            # kiwango cha msingi
            st = PITCH[k["g"]]
            if EXPRESSIVE:                                                      # hisia: sauti kubwa/ndogo na pitch ya juu/chini kama asili
                if k["rms"] > 0:
                    a = a * float(np.clip((k["rms"] / med_rms) ** 0.8, 0.5, 2.0))
                if k["f0"]:
                    st += float(np.clip(6 * np.log2(k["f0"] / med_f0[k["g"]]), -2.5, 2.5))
            items.append((s["start"], s["end"], fx(a, 1.0, st)))
        if i % 100 == 0:
            log(f"  … sauti {i}/{len(segs)}")
    voice_wav, out = f"{WORK}/voice.wav", f"{WORK}/{NAME}-sw.mp4"
    write_wav(build_track(items, total), voice_wav)
    log("🎚️ Kuchanganya na video…")
    if nov:
        mix_sep(src, nov, voice_wav, out)
    else:
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

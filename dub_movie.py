#!/usr/bin/env python3
"""dub_movie.py — Dubbing ya filamu kwa Kiswahili, bure kabisa (inakimbia kwenye GitHub Actions, CPU tu).

Mnyororo: ffmpeg (sauti) -> faster-whisper (maandishi + muda) -> NLLB-200 (tafsiri kwenda Kiswahili)
          -> MMS-TTS swh (sauti ya Kiswahili) -> ffmpeg (kuchanganya na muziki/sauti ya asili) -> mp4 mpya + .vtt -> Hugging Face

LEseni: NLLB na MMS-TTS ni CC-BY-NC 4.0 — matumizi yasiyo ya kibiashara tu, na lazima utoe sifa kwa Meta AI.
        Ukianza kupata mapato, badilisha sauti/tafsiri kwa zenye leseni ya kibiashara.

Dubbing ya kweli: Demucs huondoa sauti ya waigizaji wa asili (inabaki muziki + kelele), sauti ya Kiswahili ya kike/kiume
huchaguliwa kwa kila sentensi kutokana na sauti ya asili (pitch), na hisia za msingi (sauti kubwa/ndogo, juu/chini) huhamishwa.

Mpya: manukuu ya asili (SUBTITLE_URL au subs/<NAME>.srt) hulinganishwa na Whisper; wazungumzaji hutambuliwa na kupewa sauti hadi sita;
      hisia hutambuliwa kwa modeli; mistari mirefu hufupishwa kwa kuzungumzwa, ila manukuu (.vtt) hubeba maana kamili.

Env: SRC_URL (au njia ya faili), NAME, SRC_LANG (hiari), WHISPER_MODEL (medium), WHISPER_BEAM (3), PROMPT (majina ya wahusika), AUDIO_TRACK (0)
     SUBTITLE_URL (hiari), SUB_LANG (en), CONDENSE (1), CONDENSE_RATIO (1.35), LLM_REPO (Qwen/Qwen2.5-1.5B-Instruct)
     DIARIZE (1), SPK_THRESHOLD (0.35), VOICES_F / VOICES_M (orodha ya repo, zikitenganishwa kwa koma), EMOTION (1)
     MAX_SPEED (1.5), KEEP_ORIGINAL (0/1)
     SEPARATE (1)       kuondoa sauti ya asili kwa Demucs (nzito: saa kadhaa kwa filamu ndefu); 0 = njia ya zamani (voice-over)
     BG_SEP (0.9)       sauti ya muziki/kelele baada ya kutenganisha;  BG_VOLUME (0.25) = njia ya zamani
     VOICE_F, VOICE_M   Hugging Face repo za sauti ya kike na kiume (muundo wa MMS/VITS); chaguo-msingi: facebook/mms-tts-swh
     PITCH_F (1.5), PITCH_M (-2.5)  semitones za msingi ili sauti zitofautiane hata kama modeli ni moja
     GENDER_SPLIT (165) Hz: pitch juu ya hii = kike;  EXPRESSIVE (1) = hamisha sauti kubwa/ndogo na pitch ya asili;  VOICE_GAIN (1.0)
     HF_TOKEN, DUB_REPO (jaynesmaxtv/movie-file), DUB_PREFIX (dubbed/), SKIP_UPLOAD (0/1)

CLONING (majaribio): CLONE=1 -> kila mzungumzaji wa asili hupata sauti yake kwa Kiswahili (Chatterbox Multilingual, zero-shot, leseni MIT).
     Marejeo ya sekunde ~9 huchukuliwa kutoka kwenye sauti safi ya Demucs (work/ref_spk*.wav). Mistari inayoshindwa, au inayozidi muda
     (CLONE_DEADLINE_MIN, 250 dk tangu kuanza), hutumia MMS kama kawaida. CLONE_CFG (0.5) = uzito wa CFG. Inahitaji: pip install chatterbox-tts
     CLIP_MIN=N -> shughulikia dakika N za kwanza tu za filamu (kwa kujaribu haraka)
"""
import json, os, re, shutil, subprocess, sys, time, urllib.request, wave
import numpy as np

E = os.environ.get
SRC, NAME = E("SRC_URL", "").strip(), re.sub(r"[^A-Za-z0-9_-]+", "-", E("NAME", "movie")).strip("-") or "movie"
SRC_LANG = E("SRC_LANG", "").strip() or None
BG, MAX_SPEED, WHISPER = float(E("BG_VOLUME", "0.25")), float(E("MAX_SPEED", "1.5")), E("WHISPER_MODEL", "medium")
BEAM, PROMPT, AUDIO_TRACK = int(E("WHISPER_BEAM", "3")), E("PROMPT", "").strip(), int(E("AUDIO_TRACK", "0"))
SUB_SRC, SUB_LANG = E("SUBTITLE_URL", "").strip(), E("SUB_LANG", "en").strip() or "en"
CONDENSE, CONDENSE_RATIO, LLM_REPO = E("CONDENSE", "1") == "1", float(E("CONDENSE_RATIO", "1.35")), E("LLM_REPO", "Qwen/Qwen2.5-1.5B-Instruct")
DIARIZE, SPK_THR, SPK_REPO = E("DIARIZE", "1") == "1", float(E("SPK_THRESHOLD", "0.35")), "microsoft/wavlm-base-plus-sv"
EMOTION, EMO_REPO = E("EMOTION", "1") == "1", "superb/wav2vec2-base-superb-er"
KEEP_ORIG = E("KEEP_ORIGINAL", "0") == "1"
CLONE, CLONE_DEADLINE, CLONE_CFG, CLIP_MIN, T0 = E("CLONE", "0") == "1", float(E("CLONE_DEADLINE_MIN", "250") or 250), float(E("CLONE_CFG", "0.5") or 0.5), float(E("CLIP_MIN", "0") or 0), time.time()
EXAG = {"neu": 0.5, "hap": 0.6, "ang": 0.8, "sad": 0.35}       # 'exaggeration' ya Chatterbox kulingana na hisia
TOKEN, DUB_REPO, PREFIX = E("HF_TOKEN", "").strip(), E("DUB_REPO", "jaynesmaxtv/movie-file"), E("DUB_PREFIX", "dubbed/")
NLLB, TTS_MODEL, WORK, SR = "facebook/nllb-200-distilled-600M", "facebook/mms-tts-swh", E("WORK_DIR", "work"), 16000
SEPARATE, BG_SEP, VOICE_GAIN = E("SEPARATE", "1") == "1", float(E("BG_SEP", "0.9")), float(E("VOICE_GAIN", "1.0"))
_lst = lambda v, fb: [x.strip() for x in (E(v, "") or E(fb, "")).split(",") if x.strip()] or [TTS_MODEL]
VOICES = {"f": _lst("VOICES_F", "VOICE_F"), "m": _lst("VOICES_M", "VOICE_M")}      # sauti hadi sita: kike na kiume
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


# ---------- manukuu ya asili (SRT/VTT): kusoma, kupanga wakati, kulinganisha na Whisper ----------
def _ts(h, m, sec, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(sec) + int(ms.ljust(3, "0")[:3]) / 1000


def parse_subs(text):
    """SRT au VTT -> [(start, end, text)]. Mistari ya sauti zisizo mazungumzo ([mlio], (kicheko), ♪) huondolewa."""
    cues = []
    pat = r"(?:(\d+):)?(\d+):(\d+)[,.](\d+)\s*-->\s*(?:(\d+):)?(\d+):(\d+)[,.](\d+)[^\n]*\n(.*?)(?:\n\s*\n|\Z)"
    for m in re.finditer(pat, text.replace("\r", ""), re.S):
        g = m.groups()
        if re.search(r"[♪♫]", g[8]):                            # nyimbo: hazifanyiwi dubbing
            continue
        t = re.sub(r"<[^>]+>|\{[^}]*\}|\[[^\]]*\]|\([^)]*\)", " ", g[8])
        t = re.sub(r"(^|\s)-\s+", r"\1", re.sub(r"\s+", " ", t.replace("\n", " "))).strip()
        if re.search(r"[^\W\d_]", t):
            cues.append((_ts(g[0], g[1], g[2], g[3]), _ts(g[4], g[5], g[6], g[7]), t))
    return cues


def estimate_shift(ws, ss, span=120.0):
    """Manukuu mara nyingi yamehamishwa kwa sekunde chache (au yana tofauti ya kasi). Tafuta offset na drift zinazolinganisha vianzio vya Whisper na vya manukuu."""
    ws, ss = np.sort(np.asarray(ws, float)), np.asarray(ss, float)
    if len(ws) == 0 or len(ss) == 0:
        return 0.0, 0.0, 0.0
    if len(ws) < 5 or len(ss) < 5:                       # data kidogo mno kukadiria offset: tumia bila kuhamisha
        i = np.clip(np.searchsorted(ws, ss), 1, len(ws) - 1) if len(ws) > 1 else np.zeros(len(ss), int)
        near = np.minimum(np.abs(ws[i] - ss), np.abs(ws[np.maximum(i - 1, 0)] - ss))
        return 0.0, 0.0, float(np.mean(near < 2.0))

    def score(off, d, tol):
        t = ss * (1 + d) + off
        i = np.clip(np.searchsorted(ws, t), 1, len(ws) - 1)
        return float(np.mean(np.minimum(np.abs(ws[i] - t), np.abs(ws[i - 1] - t)) < tol))
    many = len(ws) >= 30 and len(ss) >= 30                  # drift hukadiriwa tu kukiwa na mistari ya kutosha
    drifts = np.arange(-0.005, 0.00501, 0.0005) if many else np.array([0.0])
    _, o0, d0 = max(((score(o, d, 2.0), o, d) for d in drifts for o in np.arange(-span, span + 0.01, 0.5)))
    fdr = np.arange(d0 - 0.001, d0 + 0.00101, 0.0001) if many else np.array([0.0])
    fine = [(score(o, d, 0.6), o, d) for d in fdr for o in np.arange(o0 - 3.0, o0 + 3.01, 0.1)]
    top = max(f[0] for f in fine)
    pl = [(o, d) for sc, o, d in fine if sc >= top - 0.01]            # eneo bora: chukua katikati yake
    return float(np.mean([p[0] for p in pl])), float(np.mean([p[1] for p in pl])), float(top)


def align_subs(segs, cues, wlang):
    """Maandishi kutoka manukuu (ya kuaminika zaidi), wakati kutoka Whisper. Inarudisha (segments, ripoti)."""
    off, d, rate = estimate_shift([x["start"] for x in segs], [c[0] for c in cues])
    rep = {"offset_s": round(off, 2), "drift_pct": round(d * 100, 3), "match_rate": round(rate, 2), "cues": len(cues), "whisper_segments": len(segs)}
    if rate < 0.3:
        rep["note"] = "manukuu hayalingani na sauti; yametumika Whisper pekee"
        return segs, rep
    out, used = [], set()
    for cs, ce, text in cues:
        cs, ce = cs * (1 + d) + off, ce * (1 + d) + off
        hit = [i for i, x in enumerate(segs) if x["start"] < ce - 0.1 and x["end"] > cs + 0.1]
        st, en, conf = cs, ce, None
        if hit:
            used.update(hit)
            ws, we = min(segs[i]["start"] for i in hit), max(segs[i]["end"] for i in hit)
            if we - ws <= (ce - cs) * 1.8:                      # Whisper ikiunganisha mistari mingi, tumia wakati wa manukuu
                st, en = ws, we
            conf = float(np.mean([segs[i].get("conf", 0.0) for i in hit]))
        out.append({"start": round(max(0.0, st), 2), "end": round(max(st + 0.3, en), 2), "text": text, "lang": SUB_LANG, "src": "sub", "conf": conf})
    keep = [x for i, x in enumerate(segs) if i not in used and x.get("conf", -9) > -0.6]     # usemi ambao manukuu hayana, ila Whisper ina uhakika
    out += [{**x, "lang": wlang, "src": "whisper"} for x in keep]
    out.sort(key=lambda x: x["start"])
    rep.update(from_subtitles=sum(1 for x in out if x["src"] == "sub"), from_whisper_only=len(keep))
    return out, rep


def load_wav(path):
    """Soma wav ya 16kHz mono kama safu ya float32 (Whisper inaipokea moja kwa moja, bila kutegemea maktaba ya av)."""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, "asr.wav si 16kHz mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


# ---------- hatua za AI (zinapakia models kwa uvivu) ----------
def transcribe(wav):
    from faster_whisper import WhisperModel
    model = WhisperModel(WHISPER, device="cpu", compute_type="int8")
    segs, info = model.transcribe(load_wav(wav), language=SRC_LANG, vad_filter=True, beam_size=BEAM, condition_on_previous_text=False,
                                  initial_prompt=PROMPT or None)
    out, last = [], 0
    for s in segs:
        t = s.text.strip()
        if t and s.no_speech_prob < 0.6 and s.end > s.start:
            out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": t, "conf": round(float(s.avg_logprob), 2)})
        if s.end - last >= 600:
            last = s.end; log(f"  … imefika dakika {s.end / 60:.0f}")
    return out, info.language


_NL = {}


def translate(texts, lang):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    code = NLLB_CODES.get(lang, "eng_Latn")
    if "m" not in _NL:
        _NL["m"] = AutoModelForSeq2SeqLM.from_pretrained(NLLB).eval()
    tok, mdl = AutoTokenizer.from_pretrained(NLLB, src_lang=code), _NL["m"]
    bos, out = tok.convert_tokens_to_ids("swh_Latn"), []
    for i in range(0, len(texts), 8):
        b = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=200)
        with torch.inference_mode():
            g = mdl.generate(**b, forced_bos_token_id=bos, num_beams=2, max_new_tokens=220)
        out += tok.batch_decode(g, skip_special_tokens=True)
        if (i // 8) % 25 == 0:
            log(f"  … tafsiri {min(i + 8, len(texts))}/{len(texts)}")
    return out


def translate_mixed(items):
    """items = [(maandishi, lugha)] -> tafsiri kwa mpangilio uleule (kila lugha hutafsiriwa kwa kundi lake)."""
    out = [None] * len(items)
    for lang in sorted({l for _, l in items}):
        idx = [i for i, (_, l) in enumerate(items) if l == lang]
        texts = [items[i][0] for i in idx]
        res = texts if lang == "sw" else translate(texts, lang)
        for i, r in zip(idx, res):
            out[i] = r
    return out


# ---------- kufupisha mistari mirefu (ili ilingane na muda wa mdomo); maana kamili hubaki kwenye manukuu ----------
FILLERS_SW = ["kwa kweli", "hivyo basi", "kama unavyojua", "yaani", "basi", "unajua", "tafadhali", "unaona"]


def condense_rules(text):
    t = text
    for f in FILLERS_SW:
        t = re.sub(rf"\b{f}\b[,]?\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    return t if len(t) >= 0.5 * len(text) and t else text


class Condenser:
    """LLM ndogo ya bure inafupisha sentensi ya asili (Kiingereza) kisha inatafsiriwa tena; mistari mingine hutumia kanuni rahisi."""
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(LLM_REPO)
        self.m = AutoModelForCausalLM.from_pretrained(LLM_REPO, torch_dtype=torch.float32).eval()

    def __call__(self, text, words):
        import torch
        msgs = [{"role": "system", "content": "You shorten lines of film dialogue for dubbing. Keep the meaning, names and tone. Output only the shortened line."},
                {"role": "user", "content": f"Shorten to at most {words} words:\n{text}"}]
        ids = self.tok(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt")
        with torch.inference_mode():
            g = self.m.generate(**ids, max_new_tokens=60, do_sample=False)
        r = self.tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"').split("\n")[0].strip()
        return r if r and len(r) < len(text) and len(r) >= 0.3 * len(text) else None


class Voice:
    def __init__(self, repo=TTS_MODEL):
        import torch
        from transformers import AutoTokenizer, VitsModel
        self.torch, self.tok, self.m = torch, AutoTokenizer.from_pretrained(repo), VitsModel.from_pretrained(repo).eval()
        self.sr = int(self.m.config.sampling_rate)

    def __call__(self, text, rate=1.0):
        self.m.speaking_rate = rate
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


# ---------- cloning: sauti ya kila mwigizaji kwa Kiswahili (Chatterbox Multilingual) ----------
def trim_sil(a, thr=0.02):
    """Ondoa ukimya mwanzoni/mwishoni mwa sentensi ili muda ulingane na ratiba."""
    idx = np.where(np.abs(a) > thr * (float(np.abs(a).max()) or 1.0))[0] if len(a) else []
    return a[max(0, idx[0] - int(0.03 * SR)):idx[-1] + int(0.08 * SR)] if len(idx) else a


class Cloner:
    def __init__(self, refs=None):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _ld = torch.load
        if dev == "cpu":                              # checkpoints zimehifadhiwa kwa CUDA: zipakie kwenye CPU
            torch.load = lambda *a, **k: _ld(*a, **{**k, "map_location": torch.device("cpu")})
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self.m = ChatterboxMultilingualTTS.from_pretrained(device=dev)
        finally:
            torch.load = _ld
        torch.set_num_threads(os.cpu_count() or 2)
        self.torch, self.refs, self.conds, self.sr, self.n, self.t, self.off, self.dev = torch, refs or {}, {}, int(self.m.sr), 0, 0.0, False, dev

    def selftest(self):
        """Jaribio la haraka mwanzoni (kabla ya Demucs) ili kushindwa kujulikane mapema, si baada ya saa kadhaa."""
        if getattr(self.m, "conds", None) is None:
            return f"modeli imepakiwa ({self.dev}); hakuna sauti ya msingi ya kujaribu"
        t0 = time.time()
        w = self.m.generate("Habari, karibu sana.", language_id="sw").squeeze().cpu().numpy()
        return f"modeli imepakiwa ({self.dev}); sentensi fupi: {len(w) / self.sr:.1f}s ya sauti kwa {time.time() - t0:.1f}s"

    def resample(self, w):
        try:
            import torchaudio
            return torchaudio.functional.resample(self.torch.from_numpy(w)[None], self.sr, SR)[0].numpy().astype(np.float32)
        except Exception:
            return np.interp(np.linspace(0, len(w) - 1, int(len(w) * SR / self.sr)), np.arange(len(w)), w).astype(np.float32)

    def _prep(self, s):
        if s not in self.conds:
            try:
                self.m.prepare_conditionals(self.refs[s], exaggeration=0.5)
                self.conds[s] = self.m.conds
            except Exception as e:                    # njia ya polepole: marejeo yanatolewa kila sentensi
                log(f"⚠️ cloning: njia ya haraka haikufaa kwa mzungumzaji {s} ({str(e)[:120]}); nitatumia audio_prompt_path")
                self.conds[s] = None
        return self.conds[s]

    def __call__(self, s, text, ex=0.5):
        if self.off or time.time() - T0 > CLONE_DEADLINE * 60:
            if not self.off:
                log("⏱ Muda wa cloning umeisha; mistari iliyobaki itatumia sauti za MMS")
            self.off = True
            return None
        t0, cd, kw = time.time(), self._prep(s), {}
        if cd is not None:
            self.m.conds = cd
        else:
            kw["audio_prompt_path"] = self.refs[s]
        chunks = []
        for p in [p.strip() for p in re.split(r"(?<=[.!?;:])\s+", text) if p.strip()]:
            while len(p) > 220:                       # vipande virefu: kata kwenye nafasi
                cut = p.rfind(" ", 0, 220)
                cut = cut if cut > 0 else 220
                chunks.append(p[:cut]); p = p[cut:].strip()
            chunks.append(p)
        wavs = []
        for c in chunks:
            try:
                with self.torch.inference_mode():
                    w = self.m.generate(c, language_id="sw", exaggeration=ex, cfg_weight=CLONE_CFG, **kw).squeeze().cpu().numpy().astype(np.float32)
                wavs += [trim_sil(self.resample(w)), np.zeros(int(0.12 * SR), np.float32)]
            except Exception as e:
                if self.n < 5:
                    log(f"⚠️ cloning: kipande kimeshindwa ({str(e)[:160]})")
        self.n, self.t = self.n + 1, self.t + time.time() - t0
        if self.n % 50 == 0:
            log(f"  … cloning: mistari {self.n}, wastani {self.t / self.n:.1f}s kwa mstari ({(time.time() - T0) / 60:.0f} dk tangu kuanza)")
        return np.concatenate(wavs[:-1]) if wavs else None


def build_refs(clips, spk, rms, emo, n, target=9.0):
    """Marejeo ya sauti ya kila mzungumzaji (~9s): vipande vya wazi, vya kawaida (si kelele wala kupiga kelele), kutoka kwenye sauti safi."""
    refs = {}
    for c in sorted(set(spk)):
        idx = [k for k in range(n) if spk[k] == c and clips[k] is not None and len(clips[k]) >= 2 * SR]
        if not idx:
            idx = [k for k in range(n) if spk[k] == c and clips[k] is not None and len(clips[k]) >= int(1.2 * SR)]
        if not idx:
            continue
        neu = [k for k in idx if emo[k] == "neu"]
        idx = neu if len(neu) >= 2 else idx
        med = float(np.median([rms[k] for k in idx]))
        idx.sort(key=lambda k: abs(float(np.log((rms[k] + 1e-6) / (med + 1e-6)))))
        parts, tot = [], 0.0
        for k in idx:
            parts += [clips[k], np.zeros(int(0.2 * SR), np.float32)]
            tot += len(clips[k]) / SR + 0.2
            if tot >= target:
                break
        if tot < 1.5:
            continue
        refs[c] = f"{WORK}/ref_spk{c}.wav"
        write_wav(np.concatenate(parts[:-1]), refs[c])
    return refs


# ---------- wazungumzaji (diarization) na hisia ----------
def speaker_embeddings(clips):
    """clips = [safu ya sauti 16kHz]; inarudisha safu ya embeddings (WavLM x-vector) au None kwa clip fupi mno."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe, m = AutoFeatureExtractor.from_pretrained(SPK_REPO), WavLMForXVector.from_pretrained(SPK_REPO).eval()
    out = []
    for i, c in enumerate(clips):
        if c is None or len(c) < SR:
            out.append(None); continue
        with torch.inference_mode():
            e = m(**fe(c[:6 * SR], sampling_rate=SR, return_tensors="pt")).embeddings[0].numpy()
        out.append(e / (np.linalg.norm(e) + 1e-9))
        if i % 200 == 0:
            log(f"  … wazungumzaji {i}/{len(clips)}")
    return out


def cluster_embeddings(embs, thr, kmax):
    """Kuunganisha embeddings zinazofanana (centroid, cosine) hadi umbali > thr au makundi <= kmax. Inarudisha lebo ya kundi kwa kila embedding."""
    X = np.stack(embs)
    n = len(X)
    cent, size, alive, lab = X.copy(), np.ones(n), np.ones(n, bool), list(range(n))
    while alive.sum() > 1:
        C = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
        S = C @ C.T
        S[~alive, :] = -9; S[:, ~alive] = -9; np.fill_diagonal(S, -9)
        i, j = np.unravel_index(int(np.argmax(S)), S.shape)
        if 1 - S[i, j] > thr and alive.sum() <= kmax:
            break
        cent[i] = cent[i] * size[i] + cent[j] * size[j]; size[i] += size[j]; alive[j] = False
        lab = [i if l == j else l for l in lab]
    uniq = {l: k for k, l in enumerate(sorted(set(lab), key=lambda l: -sum(1 for x in lab if x == l)))}
    return [uniq[l] for l in lab]


EMO_FX = {"ang": {"rate": 1.10, "gain": 1.35, "st": 1.0}, "hap": {"rate": 1.05, "gain": 1.15, "st": 1.5},
          "sad": {"rate": 0.92, "gain": 0.80, "st": -1.5}, "neu": {"rate": 1.0, "gain": 1.0, "st": 0.0}}


def emotions(clips):
    """Hisia ya kila sentensi ('neu','hap','ang','sad') kwa modeli ya wav2vec2; uhakika mdogo = 'neu'."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    fe, m = AutoFeatureExtractor.from_pretrained(EMO_REPO), AutoModelForAudioClassification.from_pretrained(EMO_REPO).eval()
    out = []
    for i, c in enumerate(clips):
        if c is None or len(c) < SR * 0.8:
            out.append("neu"); continue
        with torch.inference_mode():
            p = torch.softmax(m(**fe(c[:6 * SR], sampling_rate=SR, return_tensors="pt")).logits[0], -1)
        k = int(p.argmax())
        out.append(m.config.id2label[k] if float(p[k]) >= 0.5 and m.config.id2label[k] in EMO_FX else "neu")
    return out


def assign_voices(spk, gender_of, sizes):
    """kundi -> (jinsia, nambari ya sauti, pitch ya ziada). Sauti zikiisha, zinatumika tena kwa pitch tofauti."""
    out, extra = {}, [0.0, 1.5, -1.5, 3.0, -3.0]
    for g in "fm":
        cl = sorted([c for c in set(spk) if gender_of[c] == g], key=lambda c: -sizes[c])
        for k, c in enumerate(cl):
            out[c] = (g, k % len(VOICES[g]), extra[(k // len(VOICES[g])) % len(extra)])
    return out


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
            raw = run(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", src, "-map", f"0:a:{AUDIO_TRACK}", "-f", "f32le", "-ac", "2", "-ar", str(sr), "pipe:1"]).stdout
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
        graph = (f"[1:a]{vf},asplit=2[v1][v2];[0:a:{AUDIO_TRACK}]aresample=48000,aformat=channel_layouts=stereo[o];"
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


def load_subtitle():
    """Manukuu ya asili: SUBTITLE_URL (URL au njia), au faili subs/<NAME>.srt|.vtt kwenye repo."""
    for c in ([SUB_SRC] if SUB_SRC else [f"subs/{NAME}.srt", f"subs/{NAME}.vtt"]):
        if c and (c.startswith("http") or os.path.exists(c)):
            try:
                cues = parse_subs(open(download(c, f"{WORK}/sub.txt"), encoding="utf-8-sig", errors="replace").read())
                if cues:
                    log(f"📝 Manukuu ya asili: {c} ({len(cues)} mistari)")
                    return cues
                log(f"⚠️ {c}: hakuna mistari ya mazungumzo iliyosomwa")
            except Exception as e:
                log(f"⚠️ Manukuu {c} hayakusomeka: {e}")
    return []


def main():
    if not SRC:
        sys.exit("SRC_URL haipo")
    os.makedirs(WORK, exist_ok=True)
    if CLONE:                                          # kushindwa mapema badala ya baada ya saa kadhaa za Demucs
        try:
            ct = Cloner()
            log("🧬 Cloning: " + ct.selftest())
            del ct
            import gc; gc.collect()
        except Exception as e:
            sys.exit(f"⛔ Cloning haiwezi kuanza: {str(e)[:400]}. Rekebisha usakinishaji wa chatterbox-tts, au weka CLONE=0.")
    src = download(SRC, f"{WORK}/source.mp4")
    if CLIP_MIN > 0:                                   # jaribio la haraka: dakika za kwanza tu
        run(["ffmpeg", "-y", "-v", "error", "-i", src, "-t", str(CLIP_MIN * 60), "-c", "copy", f"{WORK}/clip.mp4"])
        src = f"{WORK}/clip.mp4"
        log(f"✂️ CLIP_MIN={CLIP_MIN:g}: nashughulikia dakika {CLIP_MIN:g} za kwanza tu")
    total = float(probe(src))
    log(f"▶️ {NAME}: {total / 60:.1f} dk | kutenganisha sauti: {'ndiyo' if SEPARATE else 'hapana (voice-over)'} | Whisper: {WHISPER}")
    cues = load_subtitle()
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
        run(["ffmpeg", "-y", "-v", "error", "-i", src, "-map", f"0:a:{AUDIO_TRACK}", "-ac", "1", "-ar", str(SR), asr])
    log("🎙️ Kutambua mazungumzo (Whisper)…")
    segs, lang = transcribe(asr)
    for x in segs:
        x.update(lang=lang, src="whisper")
    json.dump({"lang": lang, "segments": segs}, open(f"{WORK}/segments.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"   vipande {len(segs)}, lugha: {lang}")
    if not segs:
        sys.exit("Hakuna mazungumzo yaliyotambuliwa")
    log(f"   mazungumzo yaliyotambuliwa: {sum(x['end'] - x['start'] for x in segs) / total * 100:.0f}% ya filamu")
    if lang == "sw" and not cues and E("FORCE_SW") != "1":
        sys.exit("⛔ Filamu hii inaonekana tayari ni ya Kiswahili (DJ). Whisper hasomi Kiswahili vizuri, kwa hiyo kuirudia kungeharibu sauti. "
                 "Tumia filamu ya asili (Kiingereza n.k.), au weka FORCE_SW=1 kama una uhakika.")
    if cues:
        segs, rep = align_subs(segs, cues, lang)
        json.dump(rep, open(f"{WORK}/sub_match.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        log(f"📝 Ulinganisho wa manukuu: {rep}")
    log("🌍 Kutafsiri kwenda Kiswahili (NLLB)…")
    sw = translate_mixed([(x["text"], x["lang"]) for x in segs])
    n = len(segs)

    # --- uchambuzi wa sauti ya asili: pitch, kiwango, wazungumzaji, hisia ---
    audio = load_wav(asr)
    clips = [audio[int(x["start"] * SR):int(min(x["end"], x["start"] + 6) * SR)] for x in segs]
    clips = [c if len(c) >= SR // 2 else None for c in clips]
    del audio
    f0 = [f0_median(c) if c is not None else None for c in clips]
    rms = [float(np.sqrt(np.mean(c ** 2))) if c is not None else 0.0 for c in clips]
    prev, seg_g = "m", []
    for v in f0:
        prev = ("f" if v >= GENDER_SPLIT else "m") if v else prev
        seg_g.append(prev)
    spk = None
    if DIARIZE:
        try:
            log("👥 Kutambua wazungumzaji…")
            embs = speaker_embeddings(clips)
            ok = [k for k, e in enumerate(embs) if e is not None]
            lab = cluster_embeddings([embs[k] for k in ok], SPK_THR, len(VOICES["f"]) + len(VOICES["m"]))
            spk = [None] * n
            for k, l in zip(ok, lab):
                spk[k] = l
            for k in range(n):                              # vipande vifupi sana: chukua mzungumzaji wa jirani
                if spk[k] is None:
                    spk[k] = next((spk[q] for q in list(range(k - 1, -1, -1)) + list(range(k + 1, n)) if spk[q] is not None), 0)
        except Exception as e:
            log(f"⚠️ Kutambua wazungumzaji kumeshindwa ({str(e)[:200]}). Nitatumia kike/kiume tu.")
            spk = None
    if spk is None:
        spk = [0 if g == "f" else 1 for g in seg_g]
    gender_of = {}
    for c in set(spk):
        fs = [f0[k] for k in range(n) if spk[k] == c and f0[k]]
        gender_of[c] = ("f" if float(np.median(fs)) >= GENDER_SPLIT else "m") if fs else ("f" if spk and [seg_g[k] for k in range(n) if spk[k] == c].count("f") * 2 > list(spk).count(c) else "m")
    sizes = {c: spk.count(c) for c in set(spk)}
    vmap = assign_voices(spk, gender_of, sizes)
    emo = ["neu"] * n
    if EMOTION:
        try:
            log("🎭 Kutambua hisia…")
            emo = emotions(clips)
        except Exception as e:
            log(f"⚠️ Kutambua hisia kumeshindwa ({str(e)[:200]}). Hisia = kawaida.")
    med_f0 = {g: float(np.median([f0[k] for k in range(n) if gender_of[spk[k]] == g and f0[k]] or [150.0])) for g in "fm"}
    med_rms = float(np.median([r for r in rms if r > 0] or [1.0]))
    log(f"   wazungumzaji {len(sizes)} (kike {sum(1 for c in sizes if gender_of[c] == 'f')}, kiume {sum(1 for c in sizes if gender_of[c] == 'm')}) | hisia: "
        + ", ".join(f"{k} {emo.count(k)}" for k in EMO_FX))

    # --- kutengeneza sauti ---
    voices, cloner = {}, None
    if CLONE:
        try:
            refs = build_refs(clips, spk, rms, emo, n)
            log(f"🧬 Cloning: marejeo ya sauti {len(refs)}/{len(sizes)} wazungumzaji" + ("" if SEPARATE else " (SEPARATE imezimwa: marejeo yana muziki/kelele, ubora utashuka)"))
            if refs:
                cloner = Cloner(refs)
        except Exception as e:
            log(f"⚠️ Cloning imeshindwa kuanza ({str(e)[:300]}). Nitatumia sauti za MMS.")

    def voice_for(g, vi):
        repo = VOICES[g][vi]
        if repo not in voices:
            voices[repo] = Voice(repo)
        return voices[repo]

    def synth(i, text):
        if cloner is not None and spk[i] in cloner.refs:                       # sauti ya mwigizaji mwenyewe; MMS ikishindwa
            a = cloner(spk[i], text, EXAG[emo[i]] if EMOTION else 0.5)
            if a is not None and len(a):
                a = a / (float(np.sqrt(np.mean(a ** 2))) + 1e-9) * 0.1 * (EMO_FX[emo[i]]["gain"] if EMOTION else 1.0)
                return a * float(np.clip((rms[i] / med_rms) ** 0.8, 0.5, 2.0)) if EXPRESSIVE and rms[i] > 0 else a     # pitch haibadilishwi: sauti inabaki ya mwigizaji
        g, vi, extra = vmap[spk[i]]
        e = EMO_FX[emo[i]] if EMOTION else EMO_FX["neu"]
        a = voice_for(g, vi)(text, rate=e["rate"])
        if a is None or len(a) == 0:
            return None
        a = a / (float(np.sqrt(np.mean(a ** 2))) + 1e-9) * 0.1
        st = (PITCH[g] if VOICES[g] == [TTS_MODEL] else 0.0) + extra + e["st"]
        a = a * e["gain"]
        if EXPRESSIVE:                                                          # hisia za msingi: sauti kubwa/ndogo na pitch ya juu/chini kama asili
            if rms[i] > 0:
                a = a * float(np.clip((rms[i] / med_rms) ** 0.8, 0.5, 2.0))
            if f0[i]:
                st += float(np.clip(6 * np.log2(f0[i] / med_f0[g]), -2.5, 2.5))
        return fx(a, 1.0, st)

    log("🔊 Kutengeneza sauti za Kiswahili…")
    say, aud = list(sw), []
    for i in range(n):
        aud.append(synth(i, say[i]))
        if i % 100 == 0:
            log(f"  … sauti {i}/{n}")
    if cloner is not None:
        log(f"🧬 Cloning: mistari {cloner.n} ilijaribiwa, wastani {cloner.t / max(cloner.n, 1):.1f}s kwa mstari" + (" (muda uliisha, zilizobaki ni MMS)" if cloner.off else ""))
    avail = [max(segs[i]["end"] - segs[i]["start"], (segs[i + 1]["start"] if i + 1 < n else total) - segs[i]["start"] - 0.15, 0.3) for i in range(n)]
    need = [i for i in range(n) if aud[i] is not None and len(aud[i]) / SR / avail[i] > CONDENSE_RATIO]
    if CONDENSE and need:
        log(f"✂️ Mistari {len(need)} ni mirefu mno kwa muda wake; ninaifupisha kwa kuzungumzwa (manukuu yatabaki na maana kamili)…")
        short_src = {}
        try:
            en = [i for i in need if segs[i]["lang"] == "en"]
            cond = Condenser() if en else None
            for i in en:
                f = CONDENSE_RATIO / (len(aud[i]) / SR / avail[i])
                r = cond(segs[i]["text"], max(2, round(len(segs[i]["text"].split()) * min(f, 0.9))))
                if r:
                    short_src[i] = r
            del cond
        except Exception as e:
            log(f"⚠️ LLM ya kufupisha imeshindwa ({str(e)[:200]}); nitatumia kanuni rahisi.")
        if short_src:
            ks = list(short_src)
            for i, t in zip(ks, translate_mixed([(short_src[i], "en") for i in ks])):
                say[i] = t
        for i in need:
            if say[i] == sw[i]:
                say[i] = condense_rules(sw[i])
            if say[i] != sw[i]:
                a2 = synth(i, say[i])
                if a2 is not None and len(a2) < len(aud[i]):
                    aud[i] = a2
                else:
                    say[i] = sw[i]
        log(f"   zilizofupishwa: {sum(1 for i in need if say[i] != sw[i])}/{len(need)}")
    items = [(segs[i]["start"], segs[i]["end"], aud[i]) for i in range(n) if aud[i] is not None]
    json.dump([{**segs[i], "sw": sw[i], "sw_say": say[i], "speaker": int(spk[i]), "gender": gender_of[spk[i]], "emotion": emo[i], "f0": f0[i], "rms": round(rms[i], 4)}
               for i in range(n)], open(f"{WORK}/sw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    voice_wav, out = f"{WORK}/voice.wav", f"{WORK}/{NAME}-sw.mp4"
    write_wav(build_track(items, total), voice_wav)
    log("🎚️ Kuchanganya na video…")
    if nov:
        mix_sep(src, nov, voice_wav, out)
    else:
        mix(src, voice_wav, out)
    vtt([(segs[i]["start"], segs[i]["end"], sw[i]) for i in range(n)], f"{WORK}/{NAME}-sw.vtt")        # manukuu: maana kamili, si ile iliyofupishwa
    log(f"✅ Tayari: {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    if E("SKIP_UPLOAD") == "1" or not TOKEN:
        log("ℹ️ Kupakia kumerukwa (hakuna HF_TOKEN au SKIP_UPLOAD=1)"); return
    from huggingface_hub import HfApi
    api = HfApi(token=TOKEN)
    api.create_repo(DUB_REPO, repo_type="dataset", exist_ok=True)
    for f in (f"{NAME}-sw.mp4", f"{NAME}-sw.vtt"):
        api.upload_file(path_or_fileobj=f"{WORK}/{f}", path_in_repo=f"{PREFIX}{f}", repo_id=DUB_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=f"{WORK}/sw.json", path_in_repo=f"{PREFIX}{NAME}-sw.json", repo_id=DUB_REPO, repo_type="dataset")
    log(f"📦 https://huggingface.co/datasets/{DUB_REPO}/resolve/main/{PREFIX}{NAME}-sw.mp4")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""lipsync_check.py — kagua kama midomo na sauti vinaendana kwenye video iliyodubiwa.

Njia: kwa kila dirisha la video (kwa mfano dakika 5, sehemu 3 tofauti za filamu):
  1) mediapipe FaceMesh hupima 'shughuli ya mdomo' (kufunguka/kufungwa kwa midomo) kwenye nyuso zote zinazoonekana, mara 10 kwa sekunde;
  2) sauti hupimwa kwa nguvu (RMS) kwa vipimo vilevile vya 0.1 s;
  3) uhusiano (correlation) hutafutwa kwa kuhamisha sauti -1.0s ... +1.0s. Hamisho bora = sauti inawahi/inachelewa kiasi gani.
Ukitoa ORIG_URL (video ya asili), kipimo hicho hicho hufanywa kwa sauti ya asili kama KIWANGO CHA KULINGANISHA (maneno ya lugha nyingine
hayafanani na midomo kikamilifu, hata kwa dub nzuri; kinachojalisha ni muda wa kuanza na kuisha kwa maneno).

KIKOMO: hii ni ukaguzi wa MUDA, si kubadilisha midomo. Mdomo wa muigizaji haubadiliki kuendana na maneno ya Kiswahili (hiyo inahitaji
modeli za GPU kama Wav2Lip/LatentSync). Pia mazungumzo ya nje ya picha (msimulizi, mtu asiyeonekana) hayana mdomo wa kupimwa.

Env: SRC_URL (au njia ya faili), NAME, WINDOWS (3), WINDOW_MIN (5), ORIG_URL (hiari), HF_TOKEN, WORK_DIR (work), SELFTEST (1 = jaribio la hesabu bila video)
"""
import json, os, re, subprocess, sys, time, urllib.request, wave
import numpy as np

E = os.environ.get
SRC, NAME, ORIG = E("SRC_URL", "").strip(), re.sub(r"[^A-Za-z0-9_-]+", "-", E("NAME", "video")).strip("-") or "video", E("ORIG_URL", "").strip()
WINDOWS, WINDOW_MIN, TOKEN, WORK = int(E("WINDOWS", "3") or 3), float(E("WINDOW_MIN", "5") or 5), E("HF_TOKEN", "").strip(), E("WORK_DIR", "work")
HOP, MAXLAG = 0.1, 10          # hatua ya sekunde 0.1; hamisho hadi ±1.0 s


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} imeshindwa: {r.stderr[-400:]}")
    return r.stdout.strip()


def download(src, dest):
    if os.path.exists(src):
        return src
    req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
    if TOKEN and "huggingface.co" in src:
        req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return dest


# ---------- hesabu (haihitaji video) ----------
def audio_env(wav_path):
    """Nguvu ya sauti (log-RMS) kwa vipimo vya HOP sekunde."""
    with wave.open(wav_path) as w:
        sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    h = int(sr * HOP)
    n = len(x) // h
    return np.log1p(60 * np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(1) + 1e-12))


def mouth_activity(gap):
    """gap = ufunguko wa mdomo kwa kila kipimo (NaN = hakuna uso). Shughuli = kasi ya kubadilika kwa ufunguko ndani ya sekunde 0.5."""
    g, act = np.asarray(gap, float), np.full(len(gap), np.nan)
    for i in range(len(g)):
        seg = g[max(0, i - 2):i + 3]
        if np.isfinite(seg).sum() >= 4:
            act[i] = np.nanstd(seg)
    return act


def best_lag(mouth, audio):
    """Hamisho (kwa vipimo) la sauti linaloongeza uhusiano na mdomo. >0 = sauti inachelewa; <0 = sauti inawahi."""
    n, res = min(len(mouth), len(audio)), {}
    m, a = mouth[:n], audio[:n]
    for L in range(-MAXLAG, MAXLAG + 1):
        mm, aa = (m[:n - L], a[L:]) if L >= 0 else (m[-L:], a[:n + L])
        ok = np.isfinite(mm)
        if ok.sum() < 50 or np.std(mm[ok]) < 1e-9 or np.std(aa[ok]) < 1e-9:
            res[L] = np.nan
            continue
        res[L] = float(np.corrcoef(mm[ok], aa[ok])[0, 1])
    valid = {k: v for k, v in res.items() if np.isfinite(v)}
    if not valid:
        return None, None, None
    L = max(valid, key=valid.get)
    return L * HOP, valid[L], valid.get(0)


def verdict(face_ratio, lag, corr):
    if face_ratio < 0.15:
        return "HAITOSHI", "Nyuso chache sana kwenye sehemu hii; haiwezi kupimwa"
    if lag is None or corr is None or corr < 0.08:
        return "DHAIFU", "Midomo haionyeshi uhusiano na sauti (au mazungumzo ni ya nje ya picha)"
    if abs(lag) <= 0.25:
        return "SAWA", "Muda unalingana"
    return "TATIZO", f"Sauti {'inachelewa' if lag > 0 else 'inawahi'} kwa takriban {abs(lag) * 1000:.0f} ms"


# ---------- video ----------
def face_gaps(video, start, dur):
    import cv2
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("mediapipe haina mp.solutions; sakinisha: pip install mediapipe==0.10.21")
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    step, total, gaps, idx = max(1, round(fps * HOP)), int(dur / HOP), [], 0
    while len(gaps) < total:
        if not cap.grab():
            break
        idx += 1
        if (idx - 1) % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        h, w = frame.shape[:2]
        if h > 360:
            frame = cv2.resize(frame, (int(w * 360 / h), 360))
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        best = np.nan
        for f in (res.multi_face_landmarks or []):
            lm = f.landmark                                       # 13/14 = midomo ya ndani juu/chini; 10/152 = kipaji/kidevu (kupima ukubwa wa uso)
            face_h = abs(lm[152].y - lm[10].y) + 1e-6
            best = np.nanmax([best, abs(lm[14].y - lm[13].y) / face_h])      # uso wowote: mdomo mkubwa zaidi kwenye fremu
        gaps.append(best)
    cap.release()
    return np.array(gaps + [np.nan] * (total - len(gaps)))


def wav_of(video, start, dur, out):
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out])
    return out


def measure(video, start, dur, tag):
    gaps = face_gaps(video, start, dur)
    act = mouth_activity(gaps)
    env = audio_env(wav_of(video, start, dur, f"{WORK}/{tag}.wav"))
    n = min(len(act), len(env))
    lag, corr, c0 = best_lag(act[:n], env[:n])
    face = float(np.isfinite(gaps).mean())
    v, why = verdict(face, lag, corr)
    return {"start_min": round(start / 60, 1), "face_ratio": round(face, 2), "lag_ms": None if lag is None else int(lag * 1000),
            "corr": None if corr is None else round(corr, 3), "corr_at_zero": None if c0 is None else round(c0, 3), "verdict": v, "why": why}


def selftest():
    """Jaribio la hesabu: mdomo na sauti vya kubuni (vipindi vya kuzungumza na kimya) vyenye hamisho linalojulikana, pande zote mbili."""
    rng = np.random.default_rng(1)
    for truth in (4, -3, 0):                                              # +: sauti inachelewa vipimo 4 (0.4 s); -: inawahi
        speech, n = [], 1500
        while len(speech) < n:
            speech += [1.0] * int(rng.integers(8, 30)) + [0.0] * int(rng.integers(8, 40))
        speech = np.array(speech[:n])
        mouth = np.where(speech > 0, 0.05 + 0.04 * rng.random(n), 0.0) + 0.002 * rng.random(n)
        audio = np.roll(speech, truth) * (1 + 0.2 * rng.random(n)) + 0.02 * rng.random(n)
        lag, corr, _ = best_lag(mouth_activity(mouth), np.log1p(audio))
        log(f"selftest: hamisho lililotarajiwa {truth * HOP * 1000:+.0f} ms, lililopatikana {lag * 1000:+.0f} ms, uhusiano {corr:.2f}")
        assert abs(lag - truth * HOP) <= 0.11 and corr > 0.3, "hesabu ya hamisho si sahihi"
    assert verdict(0.05, 0.0, 0.5)[0] == "HAITOSHI" and verdict(0.9, 0.5, 0.4)[0] == "TATIZO" and verdict(0.9, 0.0, 0.4)[0] == "SAWA" and verdict(0.9, 0.0, 0.02)[0] == "DHAIFU"
    log("selftest: OK")


def main():
    if E("SELFTEST") == "1":
        return selftest()
    if not SRC:
        sys.exit("SRC_URL haipo")
    os.makedirs(WORK, exist_ok=True)
    vid = download(SRC, f"{WORK}/dubbed.mp4")
    orig = download(ORIG, f"{WORK}/orig.mp4") if ORIG else None
    dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", vid]))
    win = min(WINDOW_MIN * 60, dur)
    starts = [dur * (k + 1) / (WINDOWS + 1) - win / 2 for k in range(WINDOWS)]
    starts = [max(0.0, min(s, dur - win)) for s in starts]
    log(f"▶️ {NAME}: {dur / 60:.1f} dk | madirisha {WINDOWS} x {win / 60:.1f} dk" + (" | kiwango cha asili: ndiyo" if orig else ""))
    rows = []
    for k, s in enumerate(starts):
        log(f"🔍 Dirisha {k + 1}/{WINDOWS} (kuanzia dakika {s / 60:.1f})…")
        r = measure(vid, s, win, f"w{k}")
        if orig:
            ro = measure(orig, s, win, f"o{k}")
            r["orig_corr"], r["orig_lag_ms"] = ro["corr"], ro["lag_ms"]
        rows.append(r)
        log(f"   uso {r['face_ratio']:.0%} | hamisho {r['lag_ms']} ms | uhusiano {r['corr']} | {r['verdict']}: {r['why']}")
    ok = [r for r in rows if r["verdict"] != "HAITOSHI"]
    lags = [r["lag_ms"] for r in ok if r["lag_ms"] is not None]
    summary = {"name": NAME, "windows": rows, "median_lag_ms": int(np.median(lags)) if lags else None,
               "verdicts": {v: sum(1 for r in rows if r["verdict"] == v) for v in ("SAWA", "TATIZO", "DHAIFU", "HAITOSHI")}}
    json.dump(summary, open(f"{WORK}/lipsync-{NAME}.json", "w"), ensure_ascii=False, indent=1)
    md = [f"### Midomo na sauti: {NAME}", "", "| Dakika | Nyuso | Hamisho (ms) | Uhusiano | Asili | Hukumu |", "|---|---|---|---|---|---|"]
    md += [f"| {r['start_min']} | {r['face_ratio']:.0%} | {r['lag_ms']} | {r['corr']} | {r.get('orig_corr', '-')} | {r['verdict']}: {r['why']} |" for r in rows]
    md += ["", f"Hamisho la kati: **{summary['median_lag_ms']} ms** (+ = sauti inachelewa). Hukumu: {summary['verdicts']}",
           "", "_Ukaguzi wa muda tu. Midomo haibadiliki kuendana na maneno ya Kiswahili._"]
    open(f"{WORK}/lipsync-{NAME}.md", "w").write("\n".join(md))
    log("\n".join(md))
    if E("GITHUB_STEP_SUMMARY"):
        open(E("GITHUB_STEP_SUMMARY"), "a").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()

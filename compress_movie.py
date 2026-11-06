#!/usr/bin/env python3
"""compress_movie.py — punguza ukubwa wa filamu (mf. 3 GB -> 600 MB) kwa ffmpeg. Inakimbia kwenye GitHub Actions (CPU).

Njia: bitrate inakokotolewa kutoka muda wa filamu na ukubwa unaotaka (TARGET_MB); urefu wa picha huchaguliwa kiotomatiki ili picha isiharibike
(HEIGHT=auto), au weka namba (480, 720...) au 0 = usibadilishe. Inatumia 2-pass kwa usahihi wa ukubwa. Matokeo: mp4 moja (video ya kwanza + sauti moja).

Env: SRC_URL (au njia ya faili), NAME, TARGET_MB (600), HEIGHT (auto), CODEC (h264|h265), PRESET (fast), TWO_PASS (1), AUDIO_KBPS (96), AUDIO_TRACK (0),
     HF_TOKEN, DUB_REPO (jaynesmaxtv/movie-file), COMPRESS_PREFIX (compressed/), SKIP_UPLOAD (0/1), FORCE (1 = punguza hata kama tayari ni ndogo)
"""
import os, re, shutil, subprocess, sys, time, urllib.request

E = os.environ.get
SRC, NAME = E("SRC_URL", "").strip(), re.sub(r"[^A-Za-z0-9_-]+", "-", E("NAME", "movie")).strip("-") or "movie"
TARGET, HEIGHT, CODEC, PRESET = float(E("TARGET_MB", "600") or 600), (E("HEIGHT", "auto") or "auto").strip().lower(), (E("CODEC", "h264") or "h264").lower(), E("PRESET", "fast") or "fast"
TWO, A_KBPS, A_TRACK, FORCE = E("TWO_PASS", "1") == "1", int(E("AUDIO_KBPS", "96") or 96), int(E("AUDIO_TRACK", "0") or 0), E("FORCE") == "1"
TOKEN, REPO, PREFIX, WORK = E("HF_TOKEN", "").strip(), E("DUB_REPO", "jaynesmaxtv/movie-file"), E("COMPRESS_PREFIX", "compressed/"), E("WORK_DIR", "work")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def probe(path, args):
    r = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "csv=p=0", path], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("ffprobe imeshindwa: " + r.stderr[-400:])
    return r.stdout.strip()


def download(src, dest):
    if os.path.exists(src):
        return src
    req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
    if TOKEN and "huggingface.co" in src:
        req.add_header("Authorization", "Bearer " + TOKEN)
    got, nxt, t0 = 0, 500 << 20, time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk); got += len(chunk)
            if got >= nxt:
                log(f"  … imepakuliwa {got / 1e6:.0f} MB ({got / 1e6 / (time.time() - t0):.1f} MB/s)"); nxt += 500 << 20
    return dest


def ffmpeg(args, label, dur):
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-nostats", "-progress", "pipe:1", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0, nxt = time.time(), 0.1
    for line in p.stdout:
        if line.startswith(("out_time_us=", "out_time_ms=")):
            try:
                fr = int(line.split("=")[1]) / 1e6 / dur
            except ValueError:
                continue
            if fr >= nxt:
                log(f"  … {label}: {fr * 100:.0f}% (imepita {(time.time() - t0) / 60:.0f} dk, kasi {fr * dur / max(time.time() - t0, 1):.2f}x)"); nxt += 0.1
    err = p.stderr.read()
    if p.wait():
        raise RuntimeError(f"ffmpeg imeshindwa: {err[-600:]}")


def main():
    if not SRC:
        sys.exit("SRC_URL haipo")
    os.makedirs(WORK, exist_ok=True)
    src = download(SRC, f"{WORK}/source.mp4")
    src_mb, dur = os.path.getsize(src) / 1048576, float(probe(src, ["-show_entries", "format=duration"]))
    w, h, fr = probe(src, ["-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate"]).split(",")[:3]
    w, h, fps = int(w), int(h), (lambda a, b: a / b if b else 24.0)(*[float(x) for x in (fr.split("/") + ["1"])[:2]])
    log(f"▶️ {NAME}: {src_mb:.0f} MB, {dur / 60:.1f} dk, {w}x{h} @ {fps:.2f}fps -> lengo {TARGET:.0f} MB ({CODEC}, preset {PRESET}, {'2-pass' if TWO else '1-pass'})")
    if src_mb <= TARGET and not FORCE:
        sys.exit(f"ℹ️ Faili tayari ni ndogo ({src_mb:.0f} MB ≤ {TARGET:.0f} MB). Hakuna haja ya kupunguza (weka FORCE=1 kulazimisha).")
    v_bps = int((TARGET * 1048576 * 8 / dur - A_KBPS * 1000) * 0.96)          # 4% kwa container na tofauti za encoder
    if v_bps < 80_000:
        sys.exit(f"⛔ Lengo la {TARGET:.0f} MB ni dogo mno kwa filamu ya dakika {dur / 60:.0f} (bitrate ya video {v_bps // 1000} kbps). Ongeza target_mb.")
    bpp_min = 0.06 if CODEC == "h264" else 0.04                                    # bits kwa pixel kwa fremu: chini ya hapa picha inaharibika
    ratio = lambda hh: v_bps / (int(w * hh / h) * hh * fps)
    if HEIGHT == "auto":
        tgt = next((x for x in (1080, 720, 576, 480, 360) if x <= h and ratio(x) >= bpp_min), min(h, 360))
    else:
        tgt = int(float(HEIGHT)) or h
    tgt = min(tgt, h)
    log(f"   video {v_bps // 1000} kbps | urefu wa picha: {tgt}p (bits/pixel {ratio(tgt):.3f})" + ("" if ratio(tgt) >= bpp_min else " ⚠️ chini ya kiwango kizuri; jaribu urefu mdogo au target_mb kubwa"))
    vc = ["-map", "0:v:0", "-map", f"0:a:{A_TRACK}?", "-sn", "-dn", "-c:v", "libx264" if CODEC == "h264" else "libx265", "-preset", PRESET, "-b:v", str(v_bps),
          "-maxrate", str(int(v_bps * 1.6)), "-bufsize", str(int(v_bps * 3)), "-pix_fmt", "yuv420p"]
    if tgt < h:
        vc += ["-vf", f"scale=-2:{tgt}"]
    vc += ["-profile:v", "high"] if CODEC == "h264" else ["-tag:v", "hvc1"]
    ac, out, plog = ["-c:a", "aac", "-b:a", f"{A_KBPS}k", "-ac", "2", "-movflags", "+faststart"], f"{WORK}/{NAME}.mp4", f"{WORK}/ff2pass"
    pp = (lambda n: ["-pass", str(n), "-passlogfile", plog]) if CODEC == "h264" else (lambda n: ["-x265-params", f"pass={n}:stats={plog}.log"])
    t0 = time.time()
    if TWO:
        log("🎞️ Hatua ya 1/2 (uchambuzi)…")
        ffmpeg(["-i", src, *vc, *pp(1), "-an", "-f", "null", os.devnull], "hatua 1", dur)
        log("🎞️ Hatua ya 2/2 (kuandika)…")
        ffmpeg(["-i", src, *vc, *pp(2), *ac, out], "hatua 2", dur)
    else:
        ffmpeg(["-i", src, *vc, *ac, out], "encode", dur)
    mb = os.path.getsize(out) / 1048576
    log(f"✅ {mb:.0f} MB (lengo {TARGET:.0f}) | punguzo {src_mb / mb:.1f}x | muda {(time.time() - t0) / 60:.0f} dk" + ("  ⚠️ imezidi lengo kwa zaidi ya 10%" if mb > TARGET * 1.1 else ""))
    if os.path.abspath(src).startswith(os.path.abspath(WORK) + os.sep):          # futa nakala iliyopakuliwa tu, si faili la mtumiaji
        os.remove(src)
    url = ""
    if E("SKIP_UPLOAD") != "1" and TOKEN:
        from huggingface_hub import HfApi
        api = HfApi(token=TOKEN)
        api.create_repo(REPO, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=out, path_in_repo=f"{PREFIX}{NAME}.mp4", repo_id=REPO, repo_type="dataset")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{PREFIX}{NAME}.mp4"
        log(f"📦 {url}")
    else:
        log("ℹ️ Kupakia kumerukwa (hakuna HF_TOKEN au SKIP_UPLOAD=1)")
    for var, txt in (("GITHUB_OUTPUT", f"url={url}\nsize_mb={mb:.0f}\n"), ("GITHUB_STEP_SUMMARY", f"### {NAME}\n{src_mb:.0f} MB -> {mb:.0f} MB ({tgt}p, {CODEC})\n\n{url}\n")):
        if E(var):
            open(E(var), "a").write(txt)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Absalom converter: Google Drive -> thibitisha MP4 -> Hugging Face -> Supabase.

Inasoma absalom_episodes moja kwa moja (service role). Siri zote ziko kwenye
environment (GitHub Secrets). Hakuna siri kwenye faili hili.

Haraka: vipindi vinapakuliwa na kuthibitishwa sambamba (WORKERS), kisha
vinapakiwa kwa commit moja kwa kila CHUNK.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "jaynesmaxtv/movie-file")
HF_REPO_TYPE = os.environ.get("HF_REPO_TYPE", "dataset")  # dataset | model | space
LIMIT = int(os.environ.get("LIMIT", "80"))       # vipindi vingi zaidi kwa run moja
WORKERS = int(os.environ.get("WORKERS", "4"))    # downloads sambamba
CHUNK = int(os.environ.get("CHUNK", "4"))        # vipindi kwa commit moja ya HF
MAX_TRIES = 3
MAX_RUNTIME_MIN = int(os.environ.get("MAX_RUNTIME_MIN", "320"))
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "5"))  # acha diski ya kutosha
DRY_RUN = os.environ.get("DRY_RUN") == "1"
ONLY_SLUGS = [s.strip() for s in os.environ.get("ONLY_SLUGS", "").split(",") if s.strip()]

QUOTA_STOP = int(os.environ.get("QUOTA_STOP", "3"))  # simama baada ya makosa ya quota haya
MIN_BYTES = 200_000
MIN_SECONDS = 10.0

HF_BASE = {
    "dataset": "https://huggingface.co/datasets/",
    "model": "https://huggingface.co/",
    "space": "https://huggingface.co/spaces/",
}


def mask(text: str) -> str:
    """Ondoa siri zozote kwenye ujumbe kabla ya kuuandika."""
    for secret in (SERVICE_KEY, HF_TOKEN):
        if secret:
            text = text.replace(secret, "***")
    return text


def short(text, n: int = 400) -> str:
    return mask(" ".join(str(text).split()))[:n]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rest(method: str, path: str, **kw):
    headers = {
        "apikey": SERVICE_KEY,
        "Authorization": f"Bearer {SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    headers.update(kw.pop("headers", {}))
    r = requests.request(method, f"{SUPABASE_URL}/rest/v1/{path}", headers=headers, timeout=60, **kw)
    if not r.ok:
        raise RuntimeError(f"Supabase {r.status_code}: {short(r.text)}")
    return r.json() if r.text else None


def fetch_pending(limit: int):
    params = {
        # !inner: inachuja kwa slug ya post ikiwa ONLY_SLUGS imewekwa
        "select": "id,wp_id,position,label,drive_id,status,tries,absalom_posts!inner(slug)",
        "or": f"(status.eq.pending,and(status.eq.failed,tries.lt.{MAX_TRIES}))",
        "order": "wp_id.asc,position.asc",
        "limit": str(limit),
    }
    if ONLY_SLUGS:
        params["absalom_posts.slug"] = "in.(" + ",".join(ONLY_SLUGS) + ")"
    return rest("GET", "absalom_episodes", params=params)


def mark_done(ep, mp4_url: str):
    rows = rest(
        "PATCH",
        f"absalom_episodes?id=eq.{ep['id']}",
        json={"mp4_url": mp4_url, "status": "done", "error": None, "updated_at": now_iso()},
        headers={"Prefer": "return=representation"},
    )
    if not rows:
        print(f"  ONYO: kipindi id={ep['id']} hakikupatikana wakati wa kuandika mp4_url", flush=True)


def mark_failed(ep, reason: str):
    rest(
        "PATCH",
        f"absalom_episodes?id=eq.{ep['id']}",
        json={
            "status": "failed",
            "error": short(reason),
            "tries": int(ep.get("tries") or 0) + 1,
            "updated_at": now_iso(),
        },
    )


def mark_retry(ep, reason: str):
    """Kosa la muda (quota ya Drive): rudisha pending bila kuongeza tries."""
    rest(
        "PATCH",
        f"absalom_episodes?id=eq.{ep['id']}",
        json={"status": "pending", "error": short(reason), "updated_at": now_iso()},
    )


def is_quota(reason: str) -> bool:
    r = reason.lower()
    return "quota" in r or "haikutoa" in r


def run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def verify_mp4(path: Path):
    """Inarudisha (True, "") au (False, sababu)."""
    size = path.stat().st_size
    if size < MIN_BYTES:
        return False, f"Faili dogo sana ({size} bytes); huenda ni ukurasa wa Drive, si video"
    with open(path, "rb") as f:
        head = f.read(12)
    if head[4:8] != b"ftyp":
        return False, "Si MP4 (hakuna 'ftyp'); huenda Drive imerudisha ukurasa wa quota au ruhusa"
    try:
        p = run(
            ["ffprobe", "-v", "error", "-show_entries", "format=format_name,duration:stream=codec_type",
             "-of", "json", str(path)],
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        return False, "ffprobe imechukua muda mrefu sana"
    if p.returncode != 0:
        return False, f"ffprobe imeshindwa: {short(p.stderr)}"
    info = json.loads(p.stdout or "{}")
    fmt = info.get("format", {})
    if "mp4" not in str(fmt.get("format_name", "")):
        return False, f"Muundo si MP4: {fmt.get('format_name')}"
    if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        return False, "Hakuna video stream"
    try:
        dur = float(fmt.get("duration"))
    except (TypeError, ValueError):
        return False, "Muda wa video haujulikani"
    if dur < MIN_SECONDS:
        return False, f"Video fupi sana ({dur:.1f}s)"
    # Kusimbua mwanzo na mwisho: mwisho unagundua faili lililokatika
    for label, args in (("mwanzo", ["-t", "10"]), ("mwisho", ["-ss", str(max(0.0, dur - 8)), "-t", "8"])):
        try:
            d = run(["ffmpeg", "-v", "error", "-xerror", *args, "-i", str(path), "-f", "null", "-"], timeout=300)
        except subprocess.TimeoutExpired:
            return False, f"Kusimbua {label} kumechukua muda mrefu sana"
        if d.returncode != 0 or d.stderr.strip():
            return False, f"Kosa la kusimbua {label}: {short(d.stderr, 200)}"
    return True, ""


def download(drive_id: str, dest: Path):
    import gdown

    out = gdown.download(id=drive_id, output=str(dest), quiet=True)
    if not out or not dest.exists():
        raise RuntimeError("Drive haikutoa faili (ruhusa au quota)")


def safe_slug(slug: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._%-]+", slug)) and not slug.startswith(".") and ".." not in slug


def codec_of(path: Path, sel: str):
    """codec_name ya stream ya kwanza (v:0 au a:0); None ikiwa haipo."""
    p = run(["ffprobe", "-v", "error", "-select_streams", sel,
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)], timeout=90)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe {sel}: {short(p.stderr)}")
    out = p.stdout.strip().splitlines()
    return out[0].strip() if out and out[0].strip() else None


def convert_file(src: Path, dst: Path) -> dict:
    """Video h264 + audio aac -> copy. Video si h264 -> re-encode video na audio.
    Audio si aac (video h264) -> re-encode audio tu. Inarudisha {'vcodec','acodec','action'}."""
    vcodec = codec_of(src, "v:0")
    acodec = codec_of(src, "a:0")
    if vcodec != "h264":
        action = "reencode-video"
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-profile:v", "main",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-ac", "2",
               "-movflags", "+faststart", str(dst)]
    elif acodec not in (None, "aac"):
        action = "reencode-audio"
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
               "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-ac", "2",
               "-movflags", "+faststart", str(dst)]
    else:
        action = "copy"
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy",
               "-movflags", "+faststart", str(dst)]
    p = run(cmd, timeout=5 * 3600)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg({action}): {short(p.stderr)}")
    if codec_of(dst, "v:0") != "h264":
        raise RuntimeError("Baada ya ubadilishaji video bado si h264")
    return {"vcodec": vcodec, "acodec": acodec, "action": action}


def process(ep, tmp: Path):
    """Inapakua na kuthibitisha kipindi kimoja. Haigusi DB (hufanywa na main thread)."""
    post = ep.get("absalom_posts") or {}
    slug = post.get("slug") or ""
    tag = f"{slug}/{ep['position']}"
    if not safe_slug(slug):
        return ep, None, None, f"Slug si salama: {slug!r}"
    f = tmp / f"{ep['id']}.mp4"
    try:
        download(ep["drive_id"], f)
        ok, why = verify_mp4(f)
        if not ok:
            raise RuntimeError(why)
        out = tmp / f"{ep['id']}.out.mp4"
        info = convert_file(f, out)
        f.unlink(missing_ok=True)
        print(f"  {tag}: video={info['vcodec']} audio={info['acodec']} -> {info['action']}", flush=True)
    except Exception as e:  # noqa: BLE001
        for x in (f, tmp / f"{ep['id']}.out.mp4"):
            x.unlink(missing_ok=True)
        return ep, None, None, str(e)
    return ep, f"videos/absalom/{slug}/{ep['position']}.mp4", out, None


def commit_with_retry(api, ops, message: str):
    last = None
    for attempt in range(3):
        try:
            return api.create_commit(
                repo_id=HF_REPO, repo_type=HF_REPO_TYPE, operations=ops, commit_message=message,
            )
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(30 * (attempt + 1))
    raise last


def write_summary(lines):
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main() -> int:
    missing = [n for n, v in (("SUPABASE_URL", SUPABASE_URL), ("SUPABASE_SERVICE_ROLE_KEY", SERVICE_KEY),
                              ("HF_TOKEN", HF_TOKEN)) if not v]
    if not SUPABASE_URL or not SERVICE_KEY:
        print("SUPABASE_URL na SUPABASE_SERVICE_ROLE_KEY zinahitajika")
        return 2
    if missing and not DRY_RUN:
        print("Hazipo kwenye environment:", ", ".join(missing))
        return 2
    if HF_REPO_TYPE not in HF_BASE:
        print("HF_REPO_TYPE lazima iwe dataset, model au space")
        return 2

    started = time.time()
    deadline = started + MAX_RUNTIME_MIN * 60
    todo = fetch_pending(LIMIT)
    print(f"Vipindi vya kushughulikia: {len(todo)} (LIMIT={LIMIT}, WORKERS={WORKERS}, CHUNK={CHUNK})", flush=True)

    if DRY_RUN:
        for ep in todo:
            post = ep.get("absalom_posts") or {}
            print(f"  [{ep['status']}] {post.get('slug')}/{ep['position']}.mp4  ({ep['label']})")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=HF_TOKEN)
    done = failed = quota_hits = 0
    log = []

    for i in range(0, len(todo), CHUNK):
        if time.time() > deadline:
            print("Muda wa run umeisha; vilivyobaki vitachukuliwa run inayofuata.", flush=True)
            break
        if quota_hits >= QUOTA_STOP:
            print(f"Drive quota imefika mara {quota_hits}; nasimama, run inayofuata itaendelea.", flush=True)
            break
        free_gb = shutil.disk_usage(tempfile.gettempdir()).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"Diski ina {free_gb:.1f}GB tu; nasimama ili nisijaze runner.", flush=True)
            break

        chunk = todo[i:i + CHUNK]
        tmp = Path(tempfile.mkdtemp(prefix="absalom_"))
        try:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                results = list(pool.map(lambda ep: process(ep, tmp), chunk))

            ready = []
            for ep, path, f, reason in results:
                if reason:
                    if is_quota(reason):
                        mark_retry(ep, reason)
                        quota_hits += 1
                        log.append(f"RETRY id={ep['id']}: {short(reason, 160)}")
                        print(f"  RETRY id={ep['id']} (quota): {short(reason, 160)}", flush=True)
                    else:
                        mark_failed(ep, reason)
                        failed += 1
                        log.append(f"FAIL {ep['id']}: {short(reason, 160)}")
                        print(f"  FAIL id={ep['id']}: {short(reason, 160)}", flush=True)
                else:
                    ready.append((ep, path, f))

            if ready:
                ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f)) for _, p, f in ready]
                try:
                    commit_with_retry(api, ops, f"absalom: add {len(ready)} episode(s)")
                except Exception as e:  # noqa: BLE001
                    for ep, p, _ in ready:
                        mark_failed(ep, f"HF upload: {e}")
                        failed += 1
                        log.append(f"FAIL {p}: HF upload: {short(e, 160)}")
                    print(f"  HF upload imeshindwa: {short(e, 200)}", flush=True)
                else:
                    for ep, p, _ in ready:
                        url = f"{HF_BASE[HF_REPO_TYPE]}{HF_REPO}/resolve/main/{p}"
                        mark_done(ep, url)
                        done += 1
                        log.append(f"OK   {p}")
                        print(f"  OK {p}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    summary = (f"Imekamilika: done={done}, failed={failed}, quota_retry={quota_hits}, "
               f"dakika={(time.time() - started) / 60:.1f}")
    print(summary, flush=True)
    write_summary(["### Absalom converter", summary, "", "```", *log[-200:], "```"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

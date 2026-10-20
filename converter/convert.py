#!/usr/bin/env python3
"""movie-public2 converter: Google Drive -> thibitisha MP4 -> Hugging Face -> Supabase.

Inasoma absalom_episodes moja kwa moja (service role), si kupitia Edge Function.
Siri zote zinatoka kwenye environment (GitHub Secrets). Hakuna siri kwenye faili hili.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "jaynesmaxtv/movie-file")
HF_REPO_TYPE = os.environ.get("HF_REPO_TYPE", "dataset")  # dataset | model | space
LIMIT = int(os.environ.get("LIMIT", "60"))                # vipindi vingi zaidi kwa run moja
CHUNK = int(os.environ.get("CHUNK", "5"))                 # vipindi kwa commit moja ya HF
MAX_TRIES = 3
MAX_RUNTIME_MIN = int(os.environ.get("MAX_RUNTIME_MIN", "320"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"

MIN_BYTES = 200_000
MIN_SECONDS = 10.0

HF_BASE = {
    "dataset": "https://huggingface.co/datasets/",
    "model": "https://huggingface.co/",
    "space": "https://huggingface.co/spaces/",
}


def mask(text: str) -> str:
    """Ondoa siri zozote kwenye ujumbe kabla ya kuuandika (log au DB)."""
    for secret in (SERVICE_KEY, HF_TOKEN):
        if secret:
            text = text.replace(secret, "***")
    return text


def short(text: str, n: int = 400) -> str:
    return mask(" ".join(str(text).split()))[:n]


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
    return rest(
        "GET",
        "absalom_episodes",
        params={
            "select": "id,wp_id,position,label,drive_id,status,tries,absalom_posts(slug)",
            "or": f"(status.eq.pending,and(status.eq.failed,tries.lt.{MAX_TRIES}))",
            "order": "wp_id.asc,position.asc",
            "limit": str(limit),
        },
    )


def mark_done(ep, mp4_url: str):
    rows = rest(
        "PATCH",
        f"absalom_episodes?id=eq.{ep['id']}",
        json={
            "mp4_url": mp4_url,
            "status": "done",
            "error": None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        headers={"Prefer": "return=representation"},
    )
    if not rows:
        print(f"  ONYO: kipindi id={ep['id']} hakikupatikana wakati wa kuandika mp4_url")


def mark_failed(ep, reason: str):
    rest(
        "PATCH",
        f"absalom_episodes?id=eq.{ep['id']}",
        json={
            "status": "failed",
            "error": short(reason),
            "tries": int(ep.get("tries") or 0) + 1,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


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
    # Jaribio la kusimbua: mwanzo na mwisho (mwisho unagundua faili lililokatika)
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


def write_summary(lines):
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main() -> int:
    missing = [n for n, v in (("SUPABASE_URL", SUPABASE_URL), ("SUPABASE_SERVICE_ROLE_KEY", SERVICE_KEY),
                              ("HF_TOKEN", HF_TOKEN)) if not v]
    if missing and not DRY_RUN:
        print("Hazipo kwenye environment:", ", ".join(missing))
        return 2
    if not SUPABASE_URL or not SERVICE_KEY:
        print("SUPABASE_URL na SUPABASE_SERVICE_ROLE_KEY zinahitajika hata kwa DRY_RUN")
        return 2
    if HF_REPO_TYPE not in HF_BASE:
        print("HF_REPO_TYPE lazima iwe dataset, model au space")
        return 2

    started = time.time()
    deadline = started + MAX_RUNTIME_MIN * 60
    todo = fetch_pending(LIMIT)
    print(f"Vipindi vya kushughulikia: {len(todo)} (LIMIT={LIMIT}, CHUNK={CHUNK})")
    if DRY_RUN:
        for ep in todo:
            post = ep.get("absalom_posts") or {}
            print(f"  [{ep['status']}] {post.get('slug')}/{ep['position']}.mp4  ({ep['label']})")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=HF_TOKEN)
    done = failed = 0
    log = []

    for i in range(0, len(todo), CHUNK):
        if time.time() > deadline:
            print("Muda wa run umeisha; vilivyobaki vitachukuliwa run inayofuata.")
            break
        chunk = todo[i:i + CHUNK]
        tmp = Path(tempfile.mkdtemp(prefix="absalom_"))
        ready = []  # (ep, path_in_repo, file)
        try:
            for ep in chunk:
                post = ep.get("absalom_posts") or {}
                slug = post.get("slug") or ""
                tag = f"{slug}/{ep['position']}"
                if not safe_slug(slug):
                    mark_failed(ep, f"Slug si salama: {slug!r}")
                    failed += 1
                    log.append(f"FAIL {tag}: slug si salama")
                    continue
                f = tmp / f"{ep['id']}.mp4"
                try:
                    download(ep["drive_id"], f)
                    ok, why = verify_mp4(f)
                    if not ok:
                        raise RuntimeError(why)
                except Exception as e:  # noqa: BLE001
                    mark_failed(ep, str(e))
                    failed += 1
                    log.append(f"FAIL {tag}: {short(e, 160)}")
                    print(f"  FAIL {tag}: {short(e, 160)}")
                    f.unlink(missing_ok=True)
                    continue
                ready.append((ep, f"videos/absalom/{slug}/{ep['position']}.mp4", f))
                print(f"  verified {tag}")

            if ready:
                try:
                    api.create_commit(
                        repo_id=HF_REPO,
                        repo_type=HF_REPO_TYPE,
                        operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f)) for _, p, f in ready],
                        commit_message=f"absalom: add {len(ready)} episode(s)",
                    )
                except Exception as e:  # noqa: BLE001
                    for ep, p, _ in ready:
                        mark_failed(ep, f"HF upload: {e}")
                        failed += 1
                        log.append(f"FAIL {p}: HF upload: {short(e, 160)}")
                    print(f"  HF upload imeshindwa: {short(e, 200)}")
                else:
                    for ep, p, _ in ready:
                        url = f"{HF_BASE[HF_REPO_TYPE]}{HF_REPO}/resolve/main/{p}"
                        mark_done(ep, url)
                        done += 1
                        log.append(f"OK   {p}")
                        print(f"  OK {p}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    summary = f"Imekamilika: done={done}, failed={failed}, dakika={(time.time() - started) / 60:.1f}"
    print(summary)
    write_summary(["### Absalom converter", summary, "", "```", *log[-200:], "```"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

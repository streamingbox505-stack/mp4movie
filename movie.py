#!/usr/bin/env python3
"""ABSALOM MOVIES converter (movie.py): Google Drive -> MP4 (H.264/AAC, faststart) -> Hugging Face dataset.

Inasoma movie_parts zenye status pending (au failed na tries < 3), inabadilisha, inapakia,
kisha inaandika done. Kila sehemu inashughulikiwa kwa kujitegemea; kosa la moja halivunji mengine.
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
HF_DATASET = os.environ.get("HF_DATASET", "jaynesmaxtv/movie-file")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "20"))
MAX_TRIES = 3
MAX_RUNTIME_MIN = int(os.environ.get("MAX_RUNTIME_MIN", "270"))  # acha dakika 30 kabla ya kikomo cha 5h
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "5"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"
MIN_BYTES = 200_000


def mask(text: str) -> str:
    for secret in (SERVICE_KEY, HF_TOKEN):
        if secret:
            text = text.replace(secret, "***")
    return text


def short(text, n: int = 300) -> str:
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


def fetch_batch():
    return rest("GET", "movie_parts", params={
        "select": "id,movie_slug,part_no,drive_id,mp4_path,status,tries",
        "or": f"(status.eq.pending,and(status.eq.failed,tries.lt.{MAX_TRIES}))",
        "order": "updated_at.asc",
        "limit": str(BATCH_SIZE),
    }) or []


def patch_part(pid: int, fields: dict):
    fields = {**fields, "updated_at": now_iso()}
    rest("PATCH", f"movie_parts?id=eq.{pid}", json=fields, headers={"Prefer": "return=minimal"})


def probe(path: Path) -> dict:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=format_name,duration:stream=codec_type,codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe: {short(p.stderr)}")
    return json.loads(p.stdout or "{}")


def codecs_ok(info: dict) -> bool:
    streams = info.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"]
    auds = [s for s in streams if s.get("codec_type") == "audio"]
    if not vids:
        return False
    v = vids[0].get("codec_name")
    a = auds[0].get("codec_name") if auds else "aac"
    return v == "h264" and a == "aac"


def make_mp4(src: Path, dst: Path) -> float:
    """Inarudisha muda wa video (sekunde). Remux ikiwa codec sahihi; vinginevyo transcode."""
    info = probe(src)
    if codecs_ok(info):
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-c:a", "aac", "-movflags", "+faststart", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=5 * 3600)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg: {short(p.stderr)}")
    out = probe(dst)
    if "mp4" not in str(out.get("format", {}).get("format_name", "")):
        raise RuntimeError("Matokeo si MP4")
    if not any(s.get("codec_type") == "video" for s in out.get("streams", [])):
        raise RuntimeError("Hakuna video stream baada ya ubadilishaji")
    try:
        return float(out["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Muda wa video haujulikani")


def download(drive_id: str, dst: Path):
    import gdown

    out = gdown.download(id=drive_id, output=str(dst), quiet=True)
    if not out or not dst.exists() or dst.stat().st_size < MIN_BYTES:
        raise RuntimeError("Drive haikutoa faili la kutosha (ruhusa au quota)")


def hf_existing_size(api, path: str):
    """Inarudisha ukubwa ikiwa faili lipo kwenye HF, vinginevyo None."""
    try:
        infos = list(api.get_paths_info(HF_DATASET, [path], repo_type="dataset"))
        return infos[0].size if infos else None
    except Exception:  # noqa: BLE001
        return None


def process(part, api, tmp: Path, deadline: float):
    pid = part["id"]
    path = part["mp4_path"]
    tries = int(part.get("tries") or 0)
    patch_part(pid, {"status": "converting"})

    size = hf_existing_size(api, path)
    if size:
        patch_part(pid, {"status": "done", "size_bytes": size, "error": None})
        print(f"  ya HF: {path} ({size} bytes)", flush=True)
        return "done"

    src = tmp / f"{pid}.src"
    dst = tmp / f"{pid}.mp4"
    try:
        download(part["drive_id"], src)
        duration = make_mp4(src, dst)
        src.unlink(missing_ok=True)
        api.upload_file(path_or_fileobj=str(dst), path_in_repo=path, repo_id=HF_DATASET,
                        repo_type="dataset", commit_message=f"absalom: {path}")
        patch_part(pid, {
            "status": "done", "size_bytes": dst.stat().st_size,
            "duration_sec": int(round(duration)), "error": None,
        })
        print(f"  OK {path}", flush=True)
        return "done"
    except Exception as e:  # noqa: BLE001
        patch_part(pid, {"status": "failed", "error": short(e, 500), "tries": tries + 1})
        print(f"  FAIL {path}: {short(e, 200)}", flush=True)
        return "failed"
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


def main() -> int:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("SUPABASE_URL na SUPABASE_SERVICE_ROLE_KEY zinahitajika")
        return 2
    if not DRY_RUN and not HF_TOKEN:
        print("HF_TOKEN inahitajika")
        return 2

    deadline = time.time() + MAX_RUNTIME_MIN * 60
    batch = fetch_batch()
    print(f"Sehemu za kushughulikia: {len(batch)} (BATCH_SIZE={BATCH_SIZE})", flush=True)
    if DRY_RUN:
        for p in batch:
            print(f"  [{p['status']}] {p['mp4_path']}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    done = failed = 0
    for part in batch:
        if time.time() > deadline:
            print("Muda umeisha; zilizobaki zitachukuliwa run inayofuata.", flush=True)
            break
        free_gb = shutil.disk_usage(tempfile.gettempdir()).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"Diski ina {free_gb:.1f}GB; nasimama.", flush=True)
            break
        tmp = Path(tempfile.mkdtemp(prefix="absm_"))
        try:
            result = process(part, api, tmp, deadline)
            done += result == "done"
            failed += result == "failed"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    summary = f"Imekamilika: done={done}, failed={failed}"
    print(summary, flush=True)
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"### Absalom movies converter\n{summary}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

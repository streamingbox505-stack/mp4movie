#!/usr/bin/env python3
"""One-time repair: scan videos/ kwenye dataset, na re-encode faili ambazo video si h264.

Inafanya faili moja moja (kila faili ~100MB). Inaacha kabla ya muda wa juu ili kuepuka kikomo cha GitHub.
Env: HF_TOKEN, HF_DATASET, REPAIR_PREFIXES (koma), MAX_RUNTIME_MIN, MIN_FREE_GB, DRY_RUN=1 kuorodhesha tu.
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from convert import codec_of, convert_file, run  # noqa: F401  (run kwa ffprobe)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET = os.environ.get("HF_DATASET", "jaynesmaxtv/movie-file")
PREFIXES = [p.strip().rstrip("/") + "/" for p in os.environ.get(
    "REPAIR_PREFIXES",
    "videos/absalom/time-slip-doctor-jin-1-10dj-murphy,videos/absalom/drive-1-10dj-murphy,videos/absalom/blood-witches-1-9final",
).split(",") if p.strip()]
MAX_RUNTIME_MIN = int(os.environ.get("MAX_RUNTIME_MIN", "300"))
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "5"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=HF_TOKEN)
    files = sorted(f for f in api.list_repo_files(HF_DATASET, repo_type="dataset")
                   if f.endswith(".mp4") and any(f.startswith(p) for p in PREFIXES))
    print(f"Faili za kukagua: {len(files)} (prefixes={PREFIXES})", flush=True)
    if DRY_RUN:
        for f in files:
            print("  ", f)
        return 0

    deadline = time.time() + MAX_RUNTIME_MIN * 60
    done = skipped = failed = 0
    for path in files:
        if time.time() > deadline:
            print("Muda umeisha; endelea run inayofuata.", flush=True)
            break
        free_gb = shutil.disk_usage(tempfile.gettempdir()).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"Diski ina {free_gb:.1f}GB; nasimama.", flush=True)
            break

        tmp = Path(tempfile.mkdtemp(prefix="repair_"))
        try:
            src = Path(hf_hub_download(HF_DATASET, path, repo_type="dataset", token=HF_TOKEN, local_dir=str(tmp)))
            vcodec = codec_of(src, "v:0")
            if vcodec == "h264":
                skipped += 1
                print(f"  SKIP {path}: video=h264", flush=True)
                continue
            out = tmp / "out.mp4"
            info = convert_file(src, out)
            src.unlink(missing_ok=True)
            api.upload_file(path_or_fileobj=str(out), path_in_repo=path, repo_id=HF_DATASET,
                            repo_type="dataset", commit_message=f"repair: {path} ({info['action']})")
            done += 1
            print(f"  FIXED {path}: video={info['vcodec']} audio={info['acodec']} -> {info['action']}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {path}: {str(e)[:300]}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    summary = f"Repair: fixed={done}, skipped(h264)={skipped}, failed={failed}"
    print(summary, flush=True)
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"### Absalom repair\n{summary}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

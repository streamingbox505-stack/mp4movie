#!/usr/bin/env python3
"""Ngwele converter: Google Drive -> MP4 (h264+aac, faststart) -> Hugging Face dataset."""
import json, os, shutil, subprocess, time
import gdown, requests
from huggingface_hub import HfApi, hf_hub_download

API = os.getenv("NGWELE_API", "https://ooqvsqmsnwjqvuzbhczx.supabase.co/functions/v1/ngwele-movie-series")
REPO = os.getenv("HF_REPO", "jaynesmaxtv/movie-file")
BATCH = int(os.getenv("BATCH", "20"))
START = time.time()
STOP_NEW = START + int(os.getenv("DEADLINE_MIN", "300")) * 60   # no new item after this
HARD = START + 330 * 60                                          # ffmpeg must end before this
MAX_TRIES, TMP, FAILED = 3, "work", "ngwele_failed.json"
api = HfApi(token=os.environ["HF_TOKEN"])


def log(*a):
    print(*a, flush=True)


def fetch_todo():
    todo, offset = [], 0
    while True:
        r = requests.get(API, params={"kind": "all", "offset": offset, "limit": 50}, timeout=120)
        r.raise_for_status()
        d = r.json()
        for p in d["posts"]:
            for n, e in enumerate(p["episodes"], 1):
                if not e.get("mp4_url"):
                    todo.append({"path": f"videos/{p['slug']}-ep{n:02d}.mp4", "id": e["drive_id"],
                                 "title": f"{p['title']} {e['label']}"})
        offset += 50
        if offset >= d["total"] or not d["posts"]:
            return todo


def load_failed():
    try:
        return json.load(open(hf_hub_download(REPO, FAILED, repo_type="dataset", token=api.token, force_download=True)))
    except Exception:
        return {}


def save_failed(failed):
    api.upload_file(path_or_fileobj=json.dumps(failed, indent=1).encode(), path_in_repo=FAILED,
                    repo_id=REPO, repo_type="dataset", commit_message="ngwele failed list")


def convert(src, dst):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,pix_fmt",
                          "-of", "json", src], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"]
    v = next((x for x in s if x["codec_type"] == "video"), None)
    a = next((x for x in s if x["codec_type"] == "audio"), None)
    if not v:
        raise RuntimeError("no video stream (not a video / quota page?)")
    ok = v["codec_name"] == "h264" and v.get("pix_fmt") == "yuv420p" and a and a["codec_name"] == "aac"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-map", "0:v:0", "-map", "0:a:0?"]
    if ok:
        cmd += ["-c", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k"]
    cmd += ["-movflags", "+faststart", dst]
    subprocess.run(cmd, check=True, timeout=max(300, HARD - time.time()))
    return "remux" if ok else "re-encode"


def main():
    failed = load_failed()
    todo = [t for t in fetch_todo() if failed.get(t["path"], 0) < MAX_TRIES]
    log(f"{len(todo)} episodes to convert, batch={BATCH}")
    done = 0
    for t in todo:
        if done >= BATCH or time.time() > STOP_NEW:
            break
        if api.file_exists(REPO, t["path"], repo_type="dataset"):
            continue
        shutil.rmtree(TMP, ignore_errors=True)
        os.makedirs(TMP)
        src, dst = f"{TMP}/in.bin", f"{TMP}/out.mp4"
        try:
            log("downloading:", t["title"])
            if not gdown.download(id=t["id"], output=src, quiet=True):
                raise RuntimeError("Drive download failed (private file or quota)")
            mode = convert(src, dst)
            log(f"  {mode}, {os.path.getsize(dst) / 1e6:.0f} MB -> uploading {t['path']}")
            api.upload_file(path_or_fileobj=dst, path_in_repo=t["path"], repo_id=REPO,
                            repo_type="dataset", commit_message=f"add {t['path']}")
            if t["path"] in failed:
                del failed[t["path"]]
                save_failed(failed)
            done += 1
        except Exception as e:
            failed[t["path"]] = failed.get(t["path"], 0) + 1
            log("  FAILED:", t["title"], "->", str(e)[:200])
            save_failed(failed)
    shutil.rmtree(TMP, ignore_errors=True)
    log(f"done: {done} uploaded in {(time.time() - START) / 60:.0f} min")


if __name__ == "__main__":
    main()

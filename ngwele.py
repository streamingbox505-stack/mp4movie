#!/usr/bin/env python3
"""Ngwele converter: Google Drive -> MP4 (h264+aac, faststart) -> Hugging Face dataset."""
import json, os, re, shutil, subprocess, time
import gdown, requests
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi, hf_hub_download

API = os.getenv("NGWELE_API", "https://ooqvsqmsnwjqvuzbhczx.supabase.co/functions/v1/ngwele-movie-series")
REPO = os.getenv("HF_REPO", "jaynesmaxtv/movie-file")
BATCH = int(os.getenv("BATCH", "100"))
START = time.time()
STOP_NEW = START + int(os.getenv("DEADLINE_MIN", "300")) * 60   # no new item after this
HARD = START + 330 * 60                                          # ffmpeg must end before this
MAX_TRIES, COOLDOWN_H, MAX_BLOCKED_IN_ROW = 5, 3, 25   # quota ya Drive ni ya kila faili, si ya runner nzima
TMP, FAILED = "work", "ngwele_failed.json"
GROUP = int(os.getenv("UPLOAD_GROUP", "10"))                                   # faili ngapi kwa commit MOJA
MAX_PENDING_BYTES = int(float(os.getenv("MAX_PENDING_GB", "6")) * 1e9)         # kinga ya diski ya runner
api = HfApi(token=os.environ["HF_TOKEN"])


class DriveBlocked(Exception):
    """Drive imekataa (quota ya downloads / ruhusa)."""


def log(*a):
    print(*a, flush=True)


def hf_retry(fn, *a, **kw):
    """HF inaweza kujibu 429 (commits nyingi): subiri na ujaribu tena, usianguke."""
    for i in range(4):
        try:
            return fn(*a, **kw)
        except Exception as e:
            if "429" not in str(e) and "Too Many Requests" not in str(e):
                raise
            wait = 60 * 2 ** i
            log(f"  HF 429 - nasubiri {wait}s (jaribio {i + 1}/4)")
            time.sleep(wait)
    raise RuntimeError("HF rate limit: nimekata tamaa")


def fetch_todo():
    todo, offset = [], 0
    while True:
        r = requests.get(API, params={"kind": "all", "offset": offset, "limit": 50}, timeout=120)
        r.raise_for_status()
        d = r.json()
        for p in d["posts"]:
            for n, e in enumerate(p["episodes"], 1):
                if not e.get("mp4_url"):
                    todo.append({"path": f"videos/ngwele/{p['slug']}/ep{n:02d}.mp4", "id": e["drive_id"],
                                 "title": f"{p['title']} {e['label']}"})
        offset += 50
        if offset >= d["total"] or not d["posts"]:
            return todo


def load_failed():
    try:
        raw = json.load(open(hf_hub_download(REPO, FAILED, repo_type="dataset", token=api.token, force_download=True)))
    except Exception:
        return {}
    return {k: (v if isinstance(v, dict) else {"n": int(v), "t": 0}) for k, v in raw.items()}  # umbo la zamani (namba)


def save_failed(failed):
    try:
        hf_retry(api.upload_file, path_or_fileobj=json.dumps(failed, indent=1).encode(), path_in_repo=FAILED,
                 repo_id=REPO, repo_type="dataset", commit_message="ngwele failed list")
    except Exception as e:  # kushindwa kuhifadhi orodha hii lisiangushe run nzima
        log("could not save failed list:", str(e)[:160])


def skip_failed(failed, path):
    f = failed.get(path)
    return bool(f) and (f["n"] >= MAX_TRIES or time.time() - f["t"] < COOLDOWN_H * 3600 * f["n"])


def mark_failed(failed, path):
    f = failed.get(path, {"n": 0, "t": 0})
    failed[path] = {"n": f["n"] + 1, "t": time.time()}


def flush(pending, failed):
    """Upload faili zote zilizo tayari (`pending`) kwa commit MOJA. Inarudisha idadi iliyofanikiwa."""
    if not pending:
        return 0
    n = len(pending)
    ops = [CommitOperationAdd(path_in_repo=t["path"], path_or_fileobj=f) for t, f in pending]
    ok = 0
    try:
        log(f"uploading {n} files in ONE commit ...")
        hf_retry(api.create_commit, REPO, ops, repo_type="dataset", commit_message=f"ngwele: add {n} files")
        for t, _ in pending:
            failed.pop(t["path"], None)
        ok = n
    except Exception as e:
        log("  COMMIT FAILED:", str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__)
        for t, _ in pending:
            mark_failed(failed, t["path"])
    for _, f in pending:
        try:
            os.remove(f)
        except OSError:
            pass
    pending.clear()
    return ok


def looks_like_video(path):
    """Ukurasa wa HTML wa Drive (quota) au faili dogo sana si video."""
    if os.path.getsize(path) < 100_000:
        return False
    with open(path, "rb") as f:
        head = f.read(512).lower()
    return b"<html" not in head and b"<!doctype" not in head


def download(drive_id, dst):
    """gdown kwanza; ikishindwa jaribu kiungo cha moja kwa moja cha Drive (confirm=t)."""
    try:
        if gdown.download(id=drive_id, output=dst, quiet=True) and looks_like_video(dst):
            return
    except Exception as e:
        log("  gdown:", str(e).strip().splitlines()[0][:120])
    with requests.get("https://drive.usercontent.google.com/download", stream=True, timeout=60,
                      params={"id": drive_id, "export": "download", "confirm": "t"}) as r:
        if r.status_code != 200 or "text/html" in r.headers.get("content-type", ""):
            raise DriveBlocked("Drive imekataa (quota ya downloads au faili si ya umma)")
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if not looks_like_video(dst):
        raise DriveBlocked("Drive ilirudisha ukurasa/faili dogo, si video (quota?)")


def run_ffmpeg(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(300, HARD - time.time()))
    if p.returncode:
        lines = [l for l in p.stderr.strip().splitlines() if l.strip()]
        raise RuntimeError("ffmpeg: " + (lines[-1][:150] if lines else f"exit {p.returncode}"))


def convert(src, dst):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,pix_fmt",
                              "-of", "json", src], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        raise RuntimeError("faili si video halali (imeharibika au haikupakuliwa kikamilifu)")
    s = json.loads(out)["streams"]
    v = next((x for x in s if x["codec_type"] == "video"), None)
    a = next((x for x in s if x["codec_type"] == "audio"), None)
    if not v:
        raise RuntimeError("no video stream (not a video / quota page?)")
    maps = ["-map", "0:v:0", "-map", "0:a:0?"]
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k"]
    tail = ["-movflags", "+faststart", dst]
    even = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"]
    if v["codec_name"] == "h264" and v.get("pix_fmt") == "yuv420p" and a and a["codec_name"] == "aac":
        try:
            run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-i", src, *maps, "-c", "copy", *tail])
            return "remux"
        except RuntimeError as e:
            log("  remux failed, re-encoding:", e)
    mkv = dst + ".mkv"
    steps = [  # kila ngazi ikishindwa, inayofuata inajaribiwa (muxer ya mp4 inaweza kukataa timestamps/packets zisizo za kawaida)
        ("re-encode", [["ffmpeg", "-y", "-v", "error", "-i", src, *maps, *enc, *tail]]),
        ("re-encode (even size, genpts)", [["ffmpeg", "-y", "-v", "error", "-fflags", "+genpts", "-i", src, *maps, *even, *enc,
                                            "-avoid_negative_ts", "make_zero", "-max_muxing_queue_size", "4096", *tail]]),
        ("re-encode via mkv", [["ffmpeg", "-y", "-v", "error", "-fflags", "+genpts", "-i", src, *maps, *even, *enc, "-f", "matroska", mkv],
                               ["ffmpeg", "-y", "-v", "error", "-fflags", "+genpts", "-i", mkv, "-c", "copy",
                                "-avoid_negative_ts", "make_zero", *tail]]),
    ]
    err = None
    for name, cmds in steps:
        try:
            for c in cmds:
                run_ffmpeg(c)
            return name
        except RuntimeError as e:
            err = e
            log(f"  {name} failed: {e}")
            for p in (dst, mkv):
                if os.path.exists(p):
                    os.remove(p)
    raise err


def migrate():
    """Move old flat files videos/ngwele-<slug>-epNN.mp4 -> videos/ngwele/<slug>/epNN.mp4 (server-side, no re-upload)."""
    moves = []
    for f in api.list_repo_files(REPO, repo_type="dataset"):
        m = re.fullmatch(r"videos/(ngwele-.+)-ep(\d+)\.mp4", f)
        if m:
            moves.append((f, f"videos/ngwele/{m[1]}/ep{m[2]}.mp4"))
    for i in range(0, len(moves), 100):
        ops = []
        for old, new in moves[i:i + 100]:
            ops += [CommitOperationCopy(src_path_in_repo=old, path_in_repo=new), CommitOperationDelete(path_in_repo=old)]
        hf_retry(api.create_commit, REPO, ops, repo_type="dataset", commit_message="ngwele: move files into folders")
        log(f"moved {len(ops) // 2} files into folders")


def main():
    try:
        migrate()
    except Exception as e:
        log("migrate skipped:", str(e)[:200])
    failed = load_failed()
    before = json.dumps(failed, sort_keys=True)
    todo = [t for t in fetch_todo() if not skip_failed(failed, t["path"])]
    log(f"{len(todo)} episodes to convert, batch={BATCH}, upload group={GROUP}")
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(f"{TMP}/out")
    pending, pending_bytes, done, seq, blocked_in_row = [], 0, 0, 0, 0
    try:
        for t in todo:
            if done + len(pending) >= BATCH or time.time() > STOP_NEW:
                break
            if blocked_in_row >= MAX_BLOCKED_IN_ROW:
                log(f"Drive imezuia {blocked_in_row} mfululizo (quota). Nasimama; run inayofuata itajaribu tena.")
                break
            if api.file_exists(REPO, t["path"], repo_type="dataset"):
                continue
            cur = f"{TMP}/cur"
            shutil.rmtree(cur, ignore_errors=True)
            os.makedirs(cur)
            src, dst = f"{cur}/in.bin", f"{cur}/out.mp4"
            try:
                log("downloading:", t["title"])
                download(t["id"], src)
                blocked_in_row = 0
                mode = convert(src, dst)
                os.remove(src)
                seq += 1
                out = f"{TMP}/out/{seq:04d}.mp4"
                os.replace(dst, out)
                size = os.path.getsize(out)
                pending.append((t, out))
                pending_bytes += size
                log(f"  {mode}, {size / 1e6:.0f} MB ready ({len(pending)}/{GROUP} waiting for upload)")
                if len(pending) >= GROUP or pending_bytes >= MAX_PENDING_BYTES:
                    done += flush(pending, failed)
                    pending_bytes = 0
            except Exception as e:
                blocked_in_row = blocked_in_row + 1 if isinstance(e, DriveBlocked) else 0
                mark_failed(failed, t["path"])
                log("  FAILED:", t["title"], "->", str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__)
        done += flush(pending, failed)   # zilizobaki (chini ya GROUP) zinapakiwa pamoja mwishoni
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        if json.dumps(failed, sort_keys=True) != before:
            save_failed(failed)   # commit MOJA mwishoni
    log(f"done: {done} uploaded in {(time.time() - START) / 60:.0f} min")


if __name__ == "__main__":
    main()

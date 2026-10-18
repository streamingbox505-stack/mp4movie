#!/usr/bin/env python3
"""build_jaynestv.py — JaynesTV: live TV ya kubuni inayotumia video zako zilizopo kwenye dataset.

Inasoma MP4 zote za `videos/` kwenye dataset, inapima urefu wa kila moja (ffprobe), inapanga foleni isiyoisha, na kuandika ratiba
`jaynestv/schedule.json` kwenye dataset ile ile. App inahesabu 'sasa hivi inachezwa nini' kutoka saa (saa - epoch) mod jumla,
kwa hiyo kila mtazamaji anaona kipindi kile kile, wakati ule ule. Hakuna seva ya streaming wala gharama ya ziada.
Filamu mpya zinaongezwa mwishoni mwa foleni; zilizofutwa zinaondolewa; mpangilio uliopo hautikisiki.
Jina/DJ/picha vinachukuliwa kutoka movie-zilizotafsriwa2 (TR_URL) kama faili linalingana; vinginevyo jina linatokana na jina la faili.
Env: HF_TOKEN (kupakia) | DATA_REPO (default jaynesmaxtv/movie-file), TR_URL, MIN_SECONDS (120), WORKERS (6), LOGO_URL, LOCAL_DIR (majaribio)"""
import json, os, re, subprocess, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

E = os.environ.get
REPO, TOKEN, LOCAL, TR_URL = E("DATA_REPO", "jaynesmaxtv/movie-file"), E("HF_TOKEN", "").strip(), E("LOCAL_DIR", "").strip(), E("TR_URL", "").strip()
MIN_SEC, WORKERS, LOGO = int(E("MIN_SECONDS", "120")), int(E("WORKERS", "6")), E("LOGO_URL", "").strip()
BASE, OUT, EPOCH = f"https://huggingface.co/datasets/{REPO}/resolve/main/", "jaynestv/schedule.json", "2026-01-01T00:00:00Z"
AUTH = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def get_json(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def list_files():
    if LOCAL:
        return [{"path": "videos/" + f, "size": os.path.getsize(os.path.join(LOCAL, f))} for f in sorted(os.listdir(LOCAL)) if f.lower().endswith(".mp4")]
    from huggingface_hub import HfApi
    tree = HfApi(token=TOKEN or None).list_repo_tree(REPO, path_in_repo="videos", repo_type="dataset", recursive=False)
    return [{"path": t.path, "size": getattr(t, "size", 0)} for t in tree if t.path.lower().endswith(".mp4")]


def probe(path):
    src = os.path.join(LOCAL, os.path.basename(path)) if LOCAL else BASE + urllib.parse.quote(path)
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"]
    if TOKEN and not LOCAL:
        cmd += ["-headers", f"Authorization: Bearer {TOKEN}\r\n"]
    try:
        out = subprocess.run(cmd + [src], capture_output=True, text=True, timeout=180).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def titles():
    """faili -> {title, dj, poster} kutoka kwa filamu zako za Public (movie-zilizotafsriwa2)."""
    meta, rows = {}, (get_json(TR_URL) if TR_URL else None) or []

    def walk(o, acc):
        if isinstance(o, str):
            acc.append(o)
        elif isinstance(o, dict):
            [walk(v, acc) for v in o.values()]
        return acc
    for r in rows if isinstance(rows, list) else []:
        poster = r.get("thumb_url") or r.get("poster_url") or ""
        for u in walk(r.get("video_urls") or {}, []):
            if u.lower().endswith(".mp4"):
                meta[urllib.parse.unquote(u.rsplit("/", 1)[-1])] = {"title": r.get("title"), "dj": r.get("dj") or "", "poster": poster if poster.startswith("https://") else ""}
    return meta


def pretty(path):
    return re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", os.path.splitext(os.path.basename(path))[0])).strip().title()


def main():
    files = list_files()
    prev = get_json(BASE + OUT, AUTH) if not LOCAL else (json.load(open(E("PREV_FILE"))) if E("PREV_FILE") else None)
    pitems = {i["file"]: i for i in (prev or {}).get("items", [])}
    order = [i["file"] for i in (prev or {}).get("items", [])]
    todo = [f for f in files if f["path"] not in pitems or pitems[f["path"]].get("size") != f["size"]]
    print(f"MP4 kwenye dataset: {len(files)} | za kupima urefu: {len(todo)}", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        durs = dict(zip([f["path"] for f in todo], ex.map(lambda f: probe(f["path"]), todo)))
    meta, have = titles(), {f["path"]: f for f in files}
    names = [p for p in order if p in have] + sorted(p for p in have if p not in order)     # mpangilio uliopo + mpya mwishoni
    items, skipped = [], []
    for p in names:
        d = durs.get(p, pitems.get(p, {}).get("dur"))
        if not d or d < MIN_SEC:
            skipped.append(p); continue
        m = meta.get(os.path.basename(p), {})
        items.append({"file": p, "size": have[p]["size"], "url": BASE + urllib.parse.quote(p), "dur": round(float(d), 1),
                      "title": m.get("title") or pretty(p), "dj": m.get("dj", ""), "poster": m.get("poster", "")})
    if skipped:
        print(f"⚠️ Zimeachwa (urefu haujulikani au chini ya {MIN_SEC}s): {skipped}")
    if not items:
        sys.exit("❌ Hakuna video za kupanga.")
    sch = {"version": 1, "name": "JaynesTV", "logo": LOGO, "epoch": (prev or {}).get("epoch", EPOCH), "total": round(sum(i["dur"] for i in items), 1),
           "items": items, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    same = prev and {k: v for k, v in prev.items() if k != "updated"} == {k: v for k, v in sch.items() if k != "updated"}
    print(f"Ratiba: video {len(items)}, jumla ya saa {sch['total'] / 3600:.1f}. {'Hakuna mabadiliko.' if same else 'Imebadilika.'}")
    data = json.dumps(sch, ensure_ascii=False, separators=(",", ":")).encode()
    open("schedule.json", "wb").write(data)
    if same or LOCAL or not TOKEN:
        return 0
    from huggingface_hub import HfApi
    HfApi(token=TOKEN).upload_file(path_or_fileobj=data, path_in_repo=OUT, repo_id=REPO, repo_type="dataset", commit_message="JaynesTV schedule")
    print(f"✅ https://huggingface.co/datasets/{REPO}/blob/main/{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""build_jaynestv.py — JaynesTV: live TV ya kubuni inayotumia video zako zilizopo kwenye dataset.

Inasoma MP4 zote za `videos/` kwenye dataset, inapima urefu wa kila moja (ffprobe), inapanga foleni isiyoisha, na kuandika ratiba
`jaynestv/schedule.json` kwenye dataset ile ile. App inahesabu 'sasa hivi inachezwa nini' kutoka saa (saa - epoch) mod jumla,
kwa hiyo kila mtazamaji anaona kipindi kile kile, wakati ule ule. Hakuna seva ya streaming wala gharama ya ziada.
Filamu mpya zinaongezwa mwishoni mwa foleni; zilizofutwa zinaondolewa; mpangilio uliopo hautikisiki.
Jina/DJ/picha vinachukuliwa kutoka movie-zilizotafsriwa2 (TR_URL) kama faili linalingana; vinginevyo jina linatokana na jina la faili.
Mpya: matangazo (jaynestv/ads/), vitambulisho (jaynestv/idents/), EPG ya saa 24 (jaynestv/epg.json), hali ya LIVE inahifadhiwa.
Env: AD_EVERY (1), ADS_PER_BREAK (2) | HF_TOKEN (kupakia) | DATA_REPO (default jaynesmaxtv/movie-file), TR_URL, MIN_SECONDS (120), WORKERS (6), LOGO_URL, LOCAL_DIR (majaribio)"""
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


def list_files(folder="videos"):
    if LOCAL:
        d = LOCAL if folder == "videos" else os.path.join(LOCAL, folder)
        if not os.path.isdir(d):
            return []
        return [{"path": folder + "/" + f, "size": os.path.getsize(os.path.join(d, f))} for f in sorted(os.listdir(d)) if f.lower().endswith(".mp4")]
    from huggingface_hub import HfApi
    try:
        tree = list(HfApi(token=TOKEN or None).list_repo_tree(REPO, path_in_repo=folder, repo_type="dataset", recursive=False))
    except Exception:
        return []
    return [{"path": t.path, "size": getattr(t, "size", 0)} for t in tree if t.path.lower().endswith(".mp4")]


def probe(path):
    src = (os.path.join(LOCAL, os.path.basename(path)) if path.startswith("videos/") else os.path.join(LOCAL, path)) if LOCAL else BASE + urllib.parse.quote(path)
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


def measure(files, pitems):
    todo = [f for f in files if f["path"] not in pitems or pitems[f["path"]].get("size") != f["size"]]
    with ThreadPoolExecutor(WORKERS) as ex:
        durs = dict(zip([f["path"] for f in todo], ex.map(lambda f: probe(f["path"]), todo)))
    return {f["path"]: durs.get(f["path"], pitems.get(f["path"], {}).get("dur")) for f in files}, len(todo)


def clip(f, d, kind, title):
    return {"file": f["path"], "size": f["size"], "url": BASE + urllib.parse.quote(f["path"]), "dur": round(float(d), 1),
            "title": title, "dj": "", "poster": "", "kind": kind}


def epg(sch, hours=24):
    """Ratiba ya vipindi vya saa 24 zijazo (vipindi tu, bila matangazo)."""
    ep = datetime.fromisoformat(sch["epoch"].replace("Z", "+00:00")).timestamp()
    now = datetime.now(timezone.utc).timestamp()
    pos, t = (now - ep) % sch["total"], now - (now - ep) % sch["total"]
    out, starts, acc = [], [], 0.0
    for it in sch["items"]:
        starts.append(acc); acc += it["dur"]
    while t < now + hours * 3600 and len(out) < 500:
        for it, s in zip(sch["items"], starts):
            st = t + s
            if it.get("kind", "program") == "program" and st + it["dur"] > now and st < now + hours * 3600:
                out.append({"start": datetime.fromtimestamp(st, timezone.utc).isoformat(timespec="seconds"),
                            "end": datetime.fromtimestamp(st + it["dur"], timezone.utc).isoformat(timespec="seconds"),
                            "title": it["title"], "dj": it["dj"], "poster": it["poster"], "file": it["file"]})
        t += sch["total"]
    return {"channel": sch["name"], "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "programs": out}


def main():
    files = list_files("videos")
    ads, idents = list_files("jaynestv/ads"), list_files("jaynestv/idents")
    prev = get_json(BASE + OUT, AUTH) if not LOCAL else (json.load(open(E("PREV_FILE"))) if E("PREV_FILE") else None)
    pitems = {i["file"]: i for i in (prev or {}).get("items", [])}
    order = list(dict.fromkeys(i["file"] for i in (prev or {}).get("items", []) if i.get("kind", "program") == "program"))
    durs, n1 = measure(files, pitems)
    adur, n2 = measure(ads + idents, pitems)
    print(f"Vipindi: {len(files)} | matangazo: {len(ads)} | vitambulisho: {len(idents)} | za kupima: {n1 + n2}", flush=True)
    meta, have = titles(), {f["path"]: f for f in files}
    names = [p for p in order if p in have] + sorted(p for p in have if p not in order)
    progs, skipped = [], []
    for p in names:
        d = durs.get(p)
        if not d or d < MIN_SEC:
            skipped.append(p); continue
        m = meta.get(os.path.basename(p), {})
        progs.append({"file": p, "size": have[p]["size"], "url": BASE + urllib.parse.quote(p), "dur": round(float(d), 1),
                      "title": m.get("title") or pretty(p), "dj": m.get("dj", ""), "poster": m.get("poster", ""), "kind": "program"})
    if skipped:
        print(f"⚠️ Zimeachwa (urefu haujulikani au chini ya {MIN_SEC}s): {skipped}")
    if not progs:
        sys.exit("❌ Hakuna video za kupanga.")
    ad_ok = [clip(f, adur[f["path"]], "ad", "Tangazo") for f in ads if adur.get(f["path"])]
    id_ok = [clip(f, adur[f["path"]], "ident", "JaynesTV") for f in idents if adur.get(f["path"])]
    every, per = max(1, int(E("AD_EVERY", "1"))), max(0, int(E("ADS_PER_BREAK", "2")))
    items, ai = [], 0
    for n, p in enumerate(progs):
        if id_ok:
            items.append(id_ok[n % len(id_ok)])          # kitambulisho cha chaneli kabla ya kipindi
        items.append(p)
        if ad_ok and per and (n + 1) % every == 0:       # mapumziko ya matangazo
            for _ in range(min(per, len(ad_ok))):
                items.append(ad_ok[ai % len(ad_ok)]); ai += 1
    sch = {"version": 2, "name": "JaynesTV", "logo": LOGO, "epoch": (prev or {}).get("epoch", EPOCH), "total": round(sum(i["dur"] for i in items), 1),
           "live": (prev or {}).get("live") or {"active": False, "url": "", "title": ""},
           "items": items, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    same = prev and {k: v for k, v in prev.items() if k != "updated"} == {k: v for k, v in sch.items() if k != "updated"}
    print(f"Ratiba: vipindi {len(progs)}, vipande vyote {len(items)}, jumla ya saa {sch['total'] / 3600:.1f}. {'Hakuna mabadiliko.' if same else 'Imebadilika.'}")
    data = json.dumps(sch, ensure_ascii=False, separators=(",", ":")).encode()
    guide = json.dumps(epg(sch), ensure_ascii=False, separators=(",", ":")).encode()
    open("schedule.json", "wb").write(data); open("epg.json", "wb").write(guide)
    if LOCAL or not TOKEN:
        return 0
    from huggingface_hub import HfApi, CommitOperationAdd
    ops = [CommitOperationAdd("jaynestv/epg.json", guide)] + ([] if same else [CommitOperationAdd(OUT, data)])
    HfApi(token=TOKEN).create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message="JaynesTV schedule + EPG")
    print(f"✅ https://huggingface.co/datasets/{REPO}/blob/main/{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

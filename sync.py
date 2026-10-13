#!/usr/bin/env python3
"""sync.py — Drive -> MP4 -> HF Dataset, AUTOMATIC, bila Space (inaendeshwa na GitHub Actions).

1. Inauliza movie-public (Supabase) episodes zipi hazipo kwenye dataset ("not_in_dataset").
2. Kila moja: inapakua kutoka Drive, inabadilisha kuwa MP4 (ffmpeg) na kuipakia
   kwenye HF Dataset kama videos/<jina>.mp4 — kwa kutumia pipeline ile ile ya app.py
   (kwa hiyo jina la faili linalingana na kile movie-public inachotarajia).
Env: SYNC_URL, HF_TOKEN (lazima) | SYNC_KEY, HF_REPO, BATCH, BUDGET_MIN (hiari).
"""
import json, os, random, sys, time
import urllib.parse, urllib.request
from pathlib import Path

SYNC_URL = os.environ.get("SYNC_URL", "").strip()
SYNC_KEY = os.environ.get("SYNC_KEY", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_REPO = os.environ.get("HF_REPO", "JAYNESTV/iphone-stream").strip()
BATCH = int(os.environ.get("BATCH", "5"))          # video ngapi kwa kila run
BUDGET_MIN = int(os.environ.get("BUDGET_MIN", "300"))  # acha kuanza kazi mpya baada ya dakika hizi


def fetch_json(url, tries=3):
    headers = {"x-admin-key": SYNC_KEY} if SYNC_KEY else {}
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"  (jaribio {i + 1} limeshindwa: {e}) najaribu tena...", flush=True)
            time.sleep(5 * (i + 1))


def fetch_missing():
    """Episodes zote zisizo kwenye dataset (kurasa 10 kwa ombi, hadi done=true)."""
    missing, offset = {}, 0
    while True:
        qs = urllib.parse.urlencode(
            {"action": "all", "convert": "1", "verify": "1", "limit": 10, "offset": offset})
        data = fetch_json(f"{SYNC_URL}?{qs}")
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "movie-public imejibu kosa"))
        for m in data.get("unconverted", []):
            if m.get("reason") == "not_in_dataset" and m.get("drive_id"):
                missing.setdefault(m["drive_id"], m)
        if data.get("done"):
            return list(missing.values())
        offset = data["next_offset"]


def load_converter():
    """Pakia pipeline ya app.py bila Gradio (tunakata sehemu ya UI)."""
    src = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
    cut = src.find("# ── UI")
    ns = {"__name__": "drive_mp4"}
    exec(compile(src[:cut] if cut > 0 else src, "app.py", "exec"), ns)
    return ns["run_pipeline"]


def main():
    for k, v in (("SYNC_URL", SYNC_URL), ("HF_TOKEN", HF_TOKEN)):
        if not v:
            print(f"❌ Secret {k} haijawekwa.")
            return 2
    started = time.time()
    run_pipeline = load_converter()
    todo = fetch_missing()
    random.shuffle(todo)  # zinazoshindwa kila mara zisizuie zingine
    print(f"Zisizo kwenye dataset: {len(todo)} — nashughulikia hadi {BATCH} sasa.", flush=True)
    ok = fail = 0
    for m in todo[:BATCH]:
        if time.time() - started > BUDGET_MIN * 60:
            print("⏱️ Muda umeisha — zilizobaki zitaendelea run inayofuata.")
            break
        link = f"https://drive.google.com/file/d/{m['drive_id']}/view"
        print(f"\n▶ {m.get('slug', '?')} / {m.get('label', '?')}  ({m['drive_id']})", flush=True)
        log = urls = ""
        try:
            for log, _files, urls in run_pipeline(
                    link, "Auto (inapendekezwa)", "Asili", True, HF_REPO, HF_TOKEN):
                pass
        except Exception as e:  # noqa: BLE001
            log = f"❌ {e}"
        print("\n".join(log.splitlines()[-6:]), flush=True)
        if urls.strip():
            ok += 1
            print(f"✅ {urls.strip().splitlines()[-1]}", flush=True)
        else:
            fail += 1
            print("❌ Haikufanikiwa (angalia log hapo juu).", flush=True)
    print(f"\n🏁 Imekamilika: ✅ {ok}  ❌ {fail}  (zilizobaki: {max(0, len(todo) - ok)})")
    return 1 if (fail and not ok) else 0


if __name__ == "__main__":
    sys.exit(main())

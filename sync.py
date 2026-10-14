#!/usr/bin/env python3
"""sync.py — Drive -> MP4 -> HF Dataset, AUTOMATIC, bila Space (inaendeshwa na GitHub Actions).

1. Inauliza movie-public (Supabase) episodes zipi hazipo kwenye dataset ("not_in_dataset").
2. Kila moja: inapakua kutoka Drive, inabadilisha kuwa MP4 (ffmpeg) na kuipakia
   kwenye HF Dataset kama videos/<jina>.mp4 — kwa kutumia pipeline ile ile ya app.py
   (kwa hiyo jina la faili linalingana na kile movie-public inachotarajia).
Env: SYNC_URL, HF_TOKEN (lazima) | SYNC_KEY, HF_REPO, BATCH, BUDGET_MIN (hiari).
Kasi: `python sync.py --plan todo.json` huorodhesha zilizobaki mara moja; kisha kila shard (SHARD/SHARDS, SEED,
TODO_FILE) huchukua sehemu yake na kuzibadilisha sambamba na shard zingine (GitHub matrix).
"""
import json, os, random, sys, time
import urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYNC_URL = os.environ.get("SYNC_URL", "").strip()
SYNC_KEY = os.environ.get("SYNC_KEY", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_REPO = os.environ.get("HF_REPO", "JAYNESTV/iphone-stream").strip()
BATCH = int(os.environ.get("BATCH", "10"))         # video ngapi kwa kila shard kwa run (0 = zote hadi muda uishe)
SHARD = int(os.environ.get("SHARD", "0"))          # shard hii ni ya ngapi (0-based)
SHARDS = max(1, int(os.environ.get("SHARDS", "1")))  # shard ngapi kwa jumla
SEED = os.environ.get("SEED") or str(time.time())   # seed ileile kwa shard zote za run moja
TODO_FILE = os.environ.get("TODO_FILE", "").strip()   # orodha iliyoandaliwa na `--plan`
SCAN_WORKERS = int(os.environ.get("SCAN_WORKERS", "3"))  # kurasa ngapi za movie-public kwa pamoja
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


def _page(offset):
    qs = urllib.parse.urlencode(
        {"action": "all", "convert": "1", "verify": "1", "limit": 10, "offset": offset})
    data = fetch_json(f"{SYNC_URL}?{qs}")
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "movie-public imejibu kosa"))
    return data


def fetch_missing():
    """Episodes zote zisizo kwenye dataset. Ukurasa wa kwanza unatoa `total`; kurasa zilizobaki zinaombwa sambamba."""
    first = _page(0)
    pages = [first]
    total = int(first.get("total") or 0)
    if not first.get("done") and total > 10:
        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
            pages += list(ex.map(_page, range(10, total, 10)))
    missing = {}
    for data in pages:
        for m in data.get("unconverted", []):
            if m.get("reason") == "not_in_dataset" and m.get("drive_id"):
                missing.setdefault(m["drive_id"], m)
    return list(missing.values())


def pick_mine(todo):
    """Changanya kwa SEED ileile (kwa shard zote), kisha chukua kila SHARDS-th kuanzia SHARD."""
    todo = sorted(todo, key=lambda m: m["drive_id"])
    random.Random(SEED).shuffle(todo)  # zinazoshindwa kila mara zisizuie zingine
    mine = todo[SHARD::SHARDS]
    return mine[:BATCH] if BATCH > 0 else mine


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
    if "--plan" in sys.argv:  # hatua ya kupanga: orodhesha mara moja, hakuna ubadilishaji
        i = sys.argv.index("--plan")
        path = sys.argv[i + 1] if len(sys.argv) > i + 1 else "todo.json"
        todo = fetch_missing()
        Path(path).write_text(json.dumps(todo), encoding="utf-8")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
                f.write(f"count={len(todo)}\n")
        print(f"Zisizo kwenye dataset: {len(todo)} — zimehifadhiwa kwenye {path}", flush=True)
        return 0
    run_pipeline = load_converter()
    todo = json.loads(Path(TODO_FILE).read_text(encoding="utf-8")) if TODO_FILE and Path(TODO_FILE).exists() else fetch_missing()
    mine = pick_mine(todo)
    print(f"Zisizo kwenye dataset: {len(todo)} — shard {SHARD + 1}/{SHARDS} inashughulikia {len(mine)} sasa.", flush=True)
    ok = fail = 0
    for m in mine:
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
    print(f"\n🏁 Shard {SHARD + 1}/{SHARDS}: ✅ {ok}  ❌ {fail}  (jumla ya zilizokuwa zimebaki: {len(todo)})")
    return 1 if (fail and not ok) else 0


if __name__ == "__main__":
    sys.exit(main())

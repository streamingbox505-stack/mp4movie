"""Drive -> mp4 (ffmpeg) -> Hugging Face kwa vyanzo vinne:
  absalom, movies, ngwele  : kazi zinatoka Cloudflare (convert-api, D1) na matokeo yanarudishwa huko
  netswahili               : inasoma netswahili.com moja kwa moja (sitemap -> posts -> Drive), HF ndiyo ukweli;
                             posts zenye mp4 zinatumwa kwa tr-admin /n2/import-rows (Zilizotafsiriwa za n2)
Video hazipiti Supabase wala Cloudflare: Drive -> runner ya GitHub -> Hugging Face.
Env: CF_API, ADMIN_KEY, HF_TOKEN | BATCH=3 SOURCES=absalom,movies,ngwele,netswahili
     NETS_SINCE_DAYS=14 (0 = zote) MAIN_REPO=jaynesmaxtv/movie-file NETS_REPO=jaynesmaxtv/movie-file"""
import glob, html as htmllib, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.parse
from datetime import date, timedelta
import requests
from huggingface_hub import HfApi

CF_API = os.environ.get("CF_API", "").rstrip("/")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BATCH = int(os.environ.get("BATCH", "3"))
BATCHES = {k: int(os.environ.get("BATCH_" + k.upper(), BATCH)) for k in ("absalom", "movies", "ngwele", "netswahili")}
MAX_RUN_MIN = int(os.environ.get("MAX_RUN_MIN", "300"))  # usianze kazi mpya baada ya dakika hizi (job ya GitHub ina kikomo cha saa 6)
DEADLINE = time.time() + MAX_RUN_MIN * 60


def out_of_time():
    return time.time() > DEADLINE
SOURCES = [s.strip() for s in os.environ.get("SOURCES", "absalom,movies,ngwele,netswahili").split(",") if s.strip()]
SINCE_DAYS = int(os.environ.get("NETS_SINCE_DAYS", "14"))
NETS_REPO = os.environ.get("NETS_REPO", "jaynesmaxtv/movie-file")
api = HfApi(token=HF_TOKEN or None)


def preflight():
    """Kagua mipangilio kabla ya kupakua chochote (inazuia kupoteza muda na data kwa mipangilio mibaya)."""
    p = []
    if not HF_TOKEN:
        p.append("HF_TOKEN haijawekwa (GitHub secret HF_TOKEN)")
    if not CF_API.startswith("http"):
        p.append("CF_API haijawekwa au si URL kamili (secret CF_API = https://tr-admin.jaynes-max.workers.dev)")
    if not ADMIN_KEY:
        p.append("ADMIN_KEY haijawekwa (GitHub secret ADMIN_KEY)")
    for tool in ("ffmpeg", "ffprobe", "gdown"):
        if not shutil.which(tool):
            p.append(f"zana ya '{tool}' haipo (workflow lazima iisakinishe)")
    if p:
        print("MIPANGILIO MIBAYA:\n - " + "\n - ".join(p), file=sys.stderr)
        sys.exit(2)
UA = {"User-Agent": "JaynesMaxTV-converter/1.0"}  # Cloudflare inakataa UA za chaguo-msingi za Python (403 / 1010)


def cf(method, path, **kw):
    r = requests.request(method, CF_API + path, headers={"x-admin-key": ADMIN_KEY, **UA}, timeout=120, **kw)
    r.raise_for_status()
    return r.json()


def exists(repo, path):
    try:
        r = requests.head(f"https://huggingface.co/datasets/{repo}/resolve/main/{path}", allow_redirects=True, timeout=30)
        return r.status_code == 200
    except Exception:
        return False


def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                   "stream=codec_type,codec_name:format=duration", "-of", "json", path])
    return json.loads(out)


def make_mp4(src, dst):
    info = probe(src)
    codecs = {s["codec_type"]: s["codec_name"] for s in info["streams"]}
    dur = float(info["format"].get("duration") or 0)
    if "video" not in codecs or dur < 30:
        raise RuntimeError("faili_si_video_halali")
    if codecs["video"] == "h264" and codecs.get("audio") in (None, "aac"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c", "copy", "-movflags", "+faststart", dst]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", dst]
    subprocess.check_call(cmd)
    out_dur = float(probe(dst)["format"].get("duration") or 0)
    if abs(out_dur - dur) > 5:
        raise RuntimeError("urefu_haulingani")
    return out_dur


def download(drive_id, tmp):
    r = subprocess.run(["gdown", f"https://drive.google.com/uc?id={drive_id}", "-O", tmp + "/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()
        low = msg.lower()
        if "too many users" in low or "quota" in low:
            why = "drive_quota: faili limetazamwa/kupakuliwa mara nyingi sana, jaribu baadaye"
        elif "permission" in low or "access denied" in low or "cannot retrieve the public link" in low:
            why = "drive_private: faili si la umma (ruhusa)"
        elif "not found" in low or "404" in low:
            why = "drive_not_found: faili halipo"
        else:
            why = "gdown: " + (msg.splitlines()[-1] if msg else f"exit {r.returncode}")
        raise RuntimeError(why[:190])
    files = [f for f in glob.glob(tmp + "/*") if os.path.isfile(f)]
    if not files:
        raise RuntimeError("drive_haikupakuliwa")
    return files[0]


# jina la mp4 la netswahili = kanuni ile ile ya movie-public (slug ya jina la asili la faili la Drive)
def clean_title(n):
    i = n.rfind(".")
    b = n[:i] if (i > 0 and re.search(r"[^.]", n[:i])) else n
    b = re.sub(r"(https?://)?www\.\S+", " ", b, flags=re.I)
    b = re.sub(r"[^\w\s\-.()]", " ", b)
    b = re.sub(r"\s+", " ", b)
    b = re.sub(r"^[ .\-_]+|[ .\-_]+$", "", b)
    return (b or "video")[:80]


def mp4_name(n):
    return (re.sub(r"[^A-Za-z0-9]+", "-", clean_title(n)).strip("-").lower() or "video") + ".mp4"


def convert_and_upload(drive_id, repo, path_fn):
    """path_fn(faili_lililopakuliwa) -> hf_path. Inarudisha (hf_path, size, duration, skipped)."""
    tmp = tempfile.mkdtemp()
    try:
        src = download(drive_id, tmp)
        path = path_fn(src)
        if exists(repo, path):
            return path, None, None, True
        dst = tmp + "/out.mp4"
        dur = make_mp4(src, dst)
        for attempt in range(4):
            try:
                api.upload_file(path_or_fileobj=dst, path_in_repo=path, repo_id=repo, repo_type="dataset")
                break
            except Exception as e:  # noqa
                if attempt == 3 or not re.search(r"429|rate|timed? ?out|5\d\d", str(e), re.I):
                    raise
                print(f"  upload imekataliwa ({str(e)[:80]}), nasubiri {60 * (attempt + 1)}s...")
                time.sleep(60 * (attempt + 1))
        return path, os.path.getsize(dst), int(dur), False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_cloudflare(source):
    """Inachukua kazi kwa vipande vya 20 (kikomo cha tr-admin) hadi idadi ya BATCH ya chanzo hiki itimie."""
    target, seen, count = BATCHES[source], set(), 0
    stats = {"imefanikiwa": 0, "imeshindwa": 0, "ripoti_zimefika": 0, "ripoti_hazikufika": 0}
    try:
        while count < target and not out_of_time():
            fresh = [j for j in cf("GET", f"/jobs?source={source}&limit=20").get("jobs", [])
                     if (j.get("id") if j.get("id") is not None else j.get("key")) not in seen]
            if not fresh:
                break  # hakuna kazi mpya (zilizobaki zimeshindwa mara hii na zitajaribiwa run ijayo)
            fresh = fresh[: target - count]
            print(f"[{source}] kazi: {len(fresh)} (jumla hadi sasa {count}/{target})")
            for j in fresh:
                if out_of_time():
                    print(f"[{source}] muda umeisha, zilizobaki zinasubiri run ijayo")
                    return
                seen.add(j.get("id") if j.get("id") is not None else j.get("key"))
                count += 1
                rep = {"source": source, "id": j.get("id"), "key": j.get("key"), "hf_path": j["hf_path"]}
                try:
                    if exists(j["repo"], j["hf_path"]):
                        rep.update(ok=True)  # tayari ipo HF, usipakue tena
                    else:
                        _, size, dur, _ = convert_and_upload(j["drive_id"], j["repo"], lambda _f, p=j["hf_path"]: p)
                        rep.update(ok=True, size_bytes=size, duration_sec=dur)
                    stats["imefanikiwa"] += 1
                    print("OK  ", source, j["hf_path"])
                except Exception as e:  # noqa
                    stats["imeshindwa"] += 1
                    rep.update(ok=False, error=str(e)[:200])
                    print("FAIL", source, j["hf_path"], e, file=sys.stderr)
                try:  # ripoti ikishindwa, video bado imepakiwa HF; tutaendelea na zingine na kuonyesha kosa wazi
                    cf("POST", "/report", json=rep)
                    stats["ripoti_zimefika"] += 1
                except Exception as e:  # noqa
                    stats["ripoti_hazikufika"] += 1
                    print("RIPOTI IMESHINDWA", source, j["hf_path"], e, file=sys.stderr)
    finally:
        print(f"[{source}] muhtasari: {stats}")
    if stats["ripoti_hazikufika"]:
        raise RuntimeError(f"{stats['ripoti_hazikufika']} ripoti hazikufika D1 (angalia ADMIN_KEY / Worker / kikomo cha D1)")


# ---------- netswahili.com (port ya movie-public) ----------
SITE = "https://netswahili.com"
NUA = {"User-Agent": "Mozilla/5.0 (compatible; JaynesTV-Sync/1.0)", "Accept-Language": "sw,en;q=0.8"}
DRIVE_DL = "https://drive.usercontent.google.com/download"
DRIVE_UA = "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 Chrome/124.0 Mobile"
PAGE = 10
NAMES = {}  # drive_id -> jina la faili (mafanikio tu)


def decode(t):
    return htmllib.unescape(t)


def strip(t):
    return re.sub(r"\s+", " ", decode(re.sub(r"<[^>]*>", " ", t))).strip()


def get_text(url, data=None, headers=None, tries=3):
    err = None
    for i in range(tries):
        try:
            r = requests.request("POST" if data is not None else "GET", url, data=data,
                                 headers={**NUA, **(headers or {})}, timeout=25)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa
            err = e
            time.sleep(0.4 * (i + 1))
    raise RuntimeError(f"fetch failed: {url} — {err}")


def drive_id_of(u):
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?(?:[^#]*&)?id=|uc\?(?:[^#]*&)?id=)([\w-]{10,})", u, re.I)
    return m.group(1) if m else None


def parse_eps(html):
    html = re.sub(r"&#0?39;|&apos;", "'", html)
    out = []
    for m in re.finditer(r"startDownload\(\s*'([^']*)'\s*,\s*'([^']*)'\s*\)", html):
        if not m.group(1).strip():
            continue
        url = decode(m.group(1).strip())
        out.append({"label": decode(m.group(2).strip()), "drive_url": url, "drive_id": drive_id_of(url), "mp4_url": None})
    return out


def sitemap():
    out, seen = [], set()
    for m in re.finditer(r"<url>\s*<loc>([^<]+)</loc>(?:\s*<lastmod>([^<]*)</lastmod>)?", get_text(SITE + "/sitemap.php"), re.I):
        url = decode(m.group(1).strip())
        pm = re.search(r"/post/([^/?#]+)", url)
        if not pm or url in seen:
            continue
        seen.add(url)
        out.append({"slug": urllib.parse.unquote(pm.group(1)), "url": url, "lastmod": (m.group(2) or "").strip() or None})
    return out


def fetch_post(e):
    html = get_text(e["url"])
    h1 = re.search(r"<h1[^>]*>([\s\S]*?)</h1>", html, re.I)
    pslug = re.search(r'id="slug"[^>]*value="([^"]*)"', html, re.I)
    img = re.search(r'class="details-image"[\s\S]*?<img[^>]+src="([^"]+)"', html, re.I)
    total = re.search(r'class="count"[^>]*>\s*(\d+)\s*Total', html, re.I)
    eps = parse_eps(html)
    key = lambda x: x["label"] + "|" + x["drive_url"]  # noqa
    seen = {key(x) for x in eps}
    declared = int(total.group(1)) if total else None
    for p in range(1, 61):
        if not (len(eps) >= PAGE or (declared or 0) > len(eps)):
            break
        more = get_text(SITE + "/data/fetch_episode.php",
                        data={"starting": str(p * PAGE), "slug": decode(pslug.group(1)) if pslug else e["slug"]},
                        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                                 "X-Requested-With": "XMLHttpRequest", "Referer": e["url"], "Origin": SITE})
        fresh = [x for x in parse_eps(more) if key(x) not in seen] if more.strip() else []
        if not fresh:
            break
        seen.update(key(x) for x in fresh)
        eps += fresh
        time.sleep(0.15)
    return {"slug": e["slug"], "url": e["url"], "lastmod": e["lastmod"], "title": strip(h1.group(1)) if h1 else e["slug"],
            "poster": decode(img.group(1)) if img else None, "episodes": eps}


def disp_name(cd):
    s = re.search(r"filename\*=UTF-8''([^;]+)", cd, re.I)
    m = re.search(r'filename="?([^";]+)"?', cd, re.I)
    if s:
        return urllib.parse.unquote(s.group(1)).strip('"')
    if m:
        try:
            return m.group(1).encode("latin-1").decode("utf-8")
        except Exception:  # noqa
            return m.group(1)
    return None


def drive_name(i):
    if i in NAMES:
        return NAMES[i]
    try:
        r = requests.get(DRIVE_DL, params={"id": i, "export": "download", "confirm": "t"},
                         headers={"User-Agent": DRIVE_UA, "Range": "bytes=0-0"}, stream=True, timeout=30)
        n = disp_name(r.headers.get("content-disposition", ""))
        r.close()
        if n:
            NAMES[i] = n
        return n
    except Exception:  # noqa
        return None


def hf_url(repo, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def run_netswahili():
    entries = sitemap()
    if SINCE_DAYS:
        cutoff = (date.today() - timedelta(days=SINCE_DAYS)).isoformat()
        entries = [e for e in entries if not e["lastmod"] or e["lastmod"][:10] >= cutoff]
    print(f"[netswahili] posts za kuchunguza: {len(entries)}")
    todo, posts, reasons = [], [], {"no_drive_id": 0, "drive_unavailable": 0}
    for e in entries:
        try:
            post = fetch_post(e)
        except Exception as ex:  # noqa
            print("FAIL post", e["slug"], ex, file=sys.stderr)
            continue
        for ep in post["episodes"]:
            if not ep["drive_id"]:
                reasons["no_drive_id"] += 1
                continue
            name = drive_name(ep["drive_id"])
            if not name:
                reasons["drive_unavailable"] += 1
                continue
            path = "videos/" + mp4_name(name)
            if exists(NETS_REPO, path):
                ep["mp4_url"] = hf_url(NETS_REPO, path)
            else:
                todo.append((post, ep, path))
        posts.append(post)
    print(f"[netswahili] bado hazijageuzwa: {len(todo)} | zimeshindwa Drive/hazina kiungo: {reasons}")
    for post, ep, path in todo[:BATCHES["netswahili"]]:
        if out_of_time():
            print("[netswahili] muda umeisha, zilizobaki zinasubiri run ijayo")
            break
        try:
            convert_and_upload(ep["drive_id"], NETS_REPO, lambda _f, p=path: p)
            ep["mp4_url"] = hf_url(NETS_REPO, path)
            print("OK  ", "netswahili", path)
        except Exception as ex:  # noqa
            print("FAIL netswahili", post["slug"], ep["drive_id"], ex, file=sys.stderr)
    send = [{"slug": p["slug"], "title": p["title"], "poster": p["poster"],
             "episodes": [{"label": x["label"], "mp4_url": x["mp4_url"]} for x in p["episodes"]]}
            for p in posts if any(x["mp4_url"] for x in p["episodes"])]
    for i in range(0, len(send), 20):
        res = cf("POST", "/n2/import-rows", json={"posts": send[i:i + 20]})
        print("[netswahili] n2:", {k: res.get(k) for k in ("added", "updated", "skipped_no_mp4")})


if __name__ == "__main__":
    preflight()
    failed = False
    for s in SOURCES:
        try:
            run_netswahili() if s == "netswahili" else run_cloudflare(s)
        except Exception as e:  # noqa
            failed = True
            print(f"[{s}] imeshindwa: {e}", file=sys.stderr)
    sys.exit(1 if failed else 0)

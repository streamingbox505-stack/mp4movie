# app.py — DRIVE → MP4 CONVERTER (Space inayojitegemea, faili moja)
# Inachukua viungo vya Google Drive (au kiungo chochote cha moja kwa moja cha video), inapakua,
# inabadilisha kuwa MP4 inayocheza kwenye kila kivinjari/simu (H.264 + AAC, faststart),
# na (hiari) inapakia kwenye HF Dataset ili upate kiungo cha moja kwa moja cha kuweka kwenye app.
import os, re, sys, time, json, html, shutil, hashlib, threading, subprocess, tempfile, glob
from pathlib import Path
from urllib.parse import quote, unquote, urlparse, parse_qs

# ZeroGPU Spaces zinadai angalau function moja ya @spaces.GPU wakati wa kuanza.
# Converter hii inatumia CPU (ffmpeg) tu; function hii ni ya kimya na haifanyi chochote.
# Ukitumia "CPU basic" (inapendekezwa) kipande hiki hakina athari.
try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return None
except Exception:
    pass

WORK_DIR = os.path.join(tempfile.gettempdir(), "mp4_work")
OUT_DIR  = os.path.join(tempfile.gettempdir(), "mp4_out")
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts", ".3gp", ".mpg", ".mpeg")

_lock = threading.Lock()
_ffmpeg = {"ff": None}


class ToolError(Exception):
    pass


def human_size(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_time(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def cleanup_old(max_age_h=6):
    now = time.time()
    for base in (WORK_DIR, OUT_DIR):
        for p in glob.glob(os.path.join(base, "*")):
            try:
                if now - os.path.getmtime(p) > max_age_h * 3600:
                    shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
            except Exception:
                pass


# ── Viungo ───────────────────────────────────────────────────
def extract_drive_id(link):
    link = (link or "").strip()
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:[^#]*&)?id=)([A-Za-z0-9_-]{15,})", link)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{15,})", link) if "google.com" in link else None
    return m.group(1) if m else None


def clean_title(name):
    base = os.path.splitext(name)[0]
    base = re.sub(r"(?i)(https?://)?www\.[^\s]+", " ", base)       # ondoa anwani za tovuti kwenye jina
    base = re.sub(r"[^\w\s\-\.\(\)]", " ", base, flags=re.UNICODE)   # ondoa emoji/alama
    base = re.sub(r"\s+", " ", base).strip(" .-_")
    return (base or "video")[:80]


def slug_file(title):
    s = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()
    return s or "video"


def _filename_from_headers(r, fallback):
    cd = r.headers.get("content-disposition", "") or ""
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd, re.I)
    if m:
        return unquote(m.group(1)).strip('"')
    m = re.search(r'filename="?([^";]+)"?', cd, re.I)
    if m:
        try:
            return m.group(1).encode("latin-1").decode("utf-8")
        except Exception:
            return m.group(1)
    return fallback


def parse_confirm_form(page):
    """Ukurasa wa onyo la Drive ('haiwezi kuskani virusi') una form yenye uuid/at/confirm."""
    action = re.search(r'<form[^>]+action="([^"]+)"', page)
    if not action:
        return None, {}
    fields = {}
    for tag in re.findall(r"<input[^>]+>", page):
        n = re.search(r'name="([^"]+)"', tag)
        v = re.search(r'value="([^"]*)"', tag)
        if n:
            fields[html.unescape(n.group(1))] = html.unescape(v.group(1)) if v else ""
    return html.unescape(action.group(1)), fields


def explain_drive_page(page):
    low = page.lower()
    if "quota" in low or "too many users" in low:
        return ("Drive imezuia upakuaji (kikomo/quota). Tengeneza nakala ya faili kwenye Drive yako "
                "(Right-click > Make a copy) kisha tumia kiungo cha nakala hiyo, au jaribu baadaye.")
    if "sign in" in low or "accounts.google.com" in low or "request access" in low or "you need access" in low:
        return "Faili si la umma. Fungua Share > General access > 'Anyone with the link' kisha jaribu tena."
    if "not found" in low or "404" in low:
        return "Faili halipatikani (limefutwa au kiungo si sahihi)."
    return "Drive imerudisha ukurasa badala ya video. Hakikisha faili ni 'Anyone with the link'."


def download_gen(link, dest_dir):
    """Generator: yield ('p', ujumbe) kwa maendeleo, kisha ('file', path, jina_asili)."""
    import requests
    os.makedirs(dest_dir, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 Chrome/120 Mobile Safari/537.36"
    fid = extract_drive_id(link)
    if fid:
        url, params = "https://drive.usercontent.google.com/download", {"id": fid, "export": "download", "confirm": "t"}
        fallback = fid
    elif re.match(r"^https?://", link.strip()):
        url, params = link.strip(), None
        fallback = os.path.basename(urlparse(link).path) or "video"
    else:
        raise ToolError("Kiungo si sahihi. Weka kiungo cha Google Drive au kiungo cha moja kwa moja cha video.")
    try:
        r = s.get(url, params=params, stream=True, timeout=(15, 60), allow_redirects=True)
        if fid and "text/html" in r.headers.get("content-type", ""):
            action, fields = parse_confirm_form(r.text)
            if action and fields:
                r = s.get(action, params=fields, stream=True, timeout=(15, 60))
        if r.status_code >= 400:
            raise ToolError(f"Seva imejibu HTTP {r.status_code}.")
        if "text/html" in r.headers.get("content-type", ""):
            raise ToolError(explain_drive_page(r.text[:20000]) if fid else "Kiungo kinarudisha ukurasa wa tovuti, si faili la video.")
    except requests.RequestException as e:
        raise ToolError(f"Mtandao: {e}")
    name = _filename_from_headers(r, fallback)
    safe = re.sub(r"[^\w\.\- ]", "_", name, flags=re.UNICODE)[:120] or "video"
    if not safe.lower().endswith(VIDEO_EXTS):
        safe += ".mp4"
    path = os.path.join(dest_dir, safe)
    total = int(r.headers.get("content-length") or 0)
    got, t0, last = 0, time.time(), 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                sp = got / max(0.1, time.time() - t0)
                pct = f" ({got / total * 100:.1f}%)" if total else ""
                yield ("p", f"⬇️  {human_size(got)}" + (f" / {human_size(total)}" if total else "") + f"{pct} · {human_size(sp)}/s")
    if got < 10 * 1024:
        raise ToolError("Faili lililopakuliwa ni dogo mno (si video). Angalia ruhusa za kiungo.")
    yield ("file", path, name)


# ── ffmpeg ───────────────────────────────────────────────────
def ensure_ffmpeg():
    if _ffmpeg["ff"]:
        return _ffmpeg["ff"]
    ff = shutil.which("ffmpeg")
    if not ff:
        subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg", shell=True,
                       capture_output=True, timeout=900)
        ff = shutil.which("ffmpeg")
    if not ff:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"], capture_output=True, timeout=600)
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = None
    if not ff:
        raise ToolError("ffmpeg haikuweza kusakinishwa kwenye Space.")
    _ffmpeg["ff"] = ff
    return ff


def probe(ff, path):
    """Soma codec/urefu kutoka output ya 'ffmpeg -i' (haihitaji ffprobe)."""
    p = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=120)
    txt = p.stderr
    info = {"v": None, "a": None, "dur": 0.0, "vline": "", "w": 0, "h": 0}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if m:
        info["dur"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Stream #[^\n]*Video: ([^\n]+)", txt)
    if m:
        info["vline"] = m.group(1)
        info["v"] = m.group(1).split()[0].split("(")[0].strip(",")
        d = re.search(r"(\d{3,5})x(\d{3,5})", m.group(1))
        if d:
            info["w"], info["h"] = int(d.group(1)), int(d.group(2))
    m = re.search(r"Stream #[^\n]*Audio: (\w+)", txt)
    if m:
        info["a"] = m.group(1)
    return info


def choose_plan(mode, info):
    """Returns (video_mode, audio_mode): 'copy' au 'enc'."""
    if mode == "Remux tu (haraka, bila kupoteza ubora)":
        return "copy", "copy"
    if mode == "Re-encode kamili (H.264 + AAC)":
        return "enc", "enc"
    # Auto
    vl = info.get("vline", "")
    browser_ok_video = info.get("v") == "h264" and not re.search(r"High 10|High 4:2:2|High 4:4:4|10le|yuv422|yuv444", vl)
    v = "copy" if browser_ok_video else "enc"
    a = "copy" if info.get("a") in (None, "aac", "mp3") else "enc"
    return v, a


def build_cmd(ff, src, dst, vmode, amode, height):
    cmd = [ff, "-y", "-hide_banner", "-nostdin", "-i", src, "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn"]
    if vmode == "copy":
        cmd += ["-c:v", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
        if height:
            cmd += ["-vf", f"scale=-2:{height}:flags=lanczos"]
    if amode == "copy":
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ac", "2"]
    cmd += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", dst]
    return cmd


def convert_gen(ff, src, dst, mode, height):
    info = probe(ff, src)
    if not info["v"]:
        raise ToolError("Hakuna video ndani ya faili hili (ffmpeg haikuona video stream).")
    vmode, amode = choose_plan(mode, info)
    if height and info["h"] and info["h"] <= height:
        height = None   # usikuze video ndogo
    yield ("p", f"🎞️  Video: {info['v']} {info['w']}x{info['h']} · Audio: {info['a'] or 'hakuna'} · Urefu: {fmt_time(info['dur'])}")
    yield ("p", f"🛠️  Mpango: video={'nakili' if vmode == 'copy' else 'encode H.264'} · audio={'nakili' if amode == 'copy' else 'encode AAC'}"
                + (f" · {height}p" if (height and vmode == 'enc') else ""))
    cmd = build_cmd(ff, src, dst, vmode, amode, height if vmode == "enc" else None)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail, last, t0 = [], 0, time.time()
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                cur = int(line.split("=")[1]) / 1e6
            except Exception:
                continue
            if time.time() - last > 2:
                last = time.time()
                pct = f" ({min(99.9, cur / info['dur'] * 100):.1f}%)" if info["dur"] else ""
                speed = cur / max(0.1, time.time() - t0)
                yield ("p", f"⚙️  {fmt_time(cur)} / {fmt_time(info['dur'])}{pct} · kasi {speed:.1f}x")
        elif "=" not in line and line:
            tail.append(line)
            tail = tail[-25:]
    proc.wait()
    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 1024:
        raise ToolError("ffmpeg imeshindwa:\n" + "\n".join(tail[-12:]))
    yield ("file", dst)


# ── Upload kwenye HF Dataset (hiari) ─────────────────────────
def hf_upload(path, repo, token, name):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=path, path_in_repo=f"videos/{name}", repo_id=repo, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo}/resolve/main/videos/{quote(name)}"


# ── Pipeline kuu (generator kwa Gradio) ──────────────────────
def run_pipeline(links_text, mode, res_choice, do_upload, repo, token):
    height = {"Asili": None, "720p": 720, "480p": 480}.get(res_choice)
    links = [l.strip() for l in (links_text or "").splitlines() if l.strip()]
    logs, outputs, urls = [], [], []

    def render(extra=""):
        return "\n".join(logs) + (("\n" + extra) if extra else "")

    if not links:
        yield "❌ Weka angalau kiungo kimoja (kimoja kwa kila mstari).", None, ""
        return
    if do_upload:
        token = (token or "").strip() or os.environ.get("HF_TOKEN", "")
        repo = (repo or "").strip()
        if not token or not re.match(r"^[\w.-]+/[\w.-]+$", repo):
            yield "❌ Kwa upload weka repo kama `username/jina-la-dataset` na HF token (au weka HF_TOKEN kwenye Space Secrets).", None, ""
            return
    if not _lock.acquire(blocking=False):
        yield "⏳ Kazi nyingine inaendelea. Subiri imalize kisha ujaribu tena.", None, ""
        return
    try:
        cleanup_old()
        logs.append(f"🎬 DRIVE → MP4 · viungo: {len(links)} · mode: {mode} · ubora: {res_choice}")
        logs.append("═" * 60)
        yield render(), None, ""
        try:
            ff = ensure_ffmpeg()
        except ToolError as e:
            logs.append(f"❌ {e}")
            yield render(), None, ""
            return
        for i, link in enumerate(links, 1):
            logs.append(f"\n[{i}/{len(links)}] {link[:90]}")
            job = os.path.join(WORK_DIR, hashlib.md5(f"{link}{time.time()}".encode()).hexdigest()[:8])
            try:
                src = orig = None
                for ev in download_gen(link, job):
                    if ev[0] == "p":
                        yield render(ev[1]), outputs or None, "\n".join(urls)
                    else:
                        src, orig = ev[1], ev[2]
                logs.append(f"✅ Imepakuliwa: {os.path.basename(src)} ({human_size(os.path.getsize(src))})")
                title = clean_title(orig)
                dst = os.path.join(OUT_DIR, f"{slug_file(title)}.mp4")
                if os.path.exists(dst):
                    if dst in outputs:  # jina limegongana ndani ya kundi hili moja
                        dst = os.path.join(OUT_DIR, f"{slug_file(title)}-{i}.mp4")
                    else:  # faili la zamani la kikao kilichopita: liandike upya, jina libaki sahihi
                        os.remove(dst)
                out = None
                for ev in convert_gen(ff, src, dst, mode, height):
                    if ev[0] == "p":
                        yield render(ev[1]), outputs or None, "\n".join(urls)
                    else:
                        out = ev[1]
                logs.append(f"✅ MP4 tayari: {os.path.basename(out)} ({human_size(os.path.getsize(out))})")
                outputs.append(out)
                if do_upload:
                    logs.append("☁️  Inapakia kwenye HF Dataset...")
                    yield render(), outputs, "\n".join(urls)
                    u = hf_upload(out, repo, token, os.path.basename(out))
                    urls.append(u)
                    logs.append(f"🔗 {u}")
            except ToolError as e:
                logs.append(f"❌ {e}")
            except Exception as e:
                logs.append(f"❌ Hitilafu: {e}")
            finally:
                shutil.rmtree(job, ignore_errors=True)
            yield render(), outputs or None, "\n".join(urls)
        logs.append("\n" + "═" * 60)
        logs.append(f"🏁 Imekamilika: {len(outputs)}/{len(links)} zimefanikiwa.")
        if outputs:
            logs.append("Pakua MP4 kwenye kisanduku cha Download. Zinahifadhiwa kwa masaa 6 tu kwenye Space.")
        yield render(), outputs or None, "\n".join(urls)
    finally:
        _lock.release()


# ── UI ───────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background:#060810 !important; color:#c8ffd4 !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
textarea, input[type="text"], input[type="password"] { background:#090d12 !important; color:#00ff88 !important; border:1px solid #1e4a3a !important; }
#log textarea { background:#040710 !important; color:#b8ffcc !important; font-family: monospace !important; font-size:0.85em !important; }
button.primary { background: linear-gradient(135deg,#00cc66,#0099ff) !important; color:#000 !important; border:none !important; font-weight:700 !important; }
footer { display:none !important; }
"""


def build_ui():
    import gradio as gr
    with gr.Blocks(title="DRIVE → MP4", css=CSS) as demo:
        gr.Markdown(
            "# 🎬 DRIVE → MP4\n"
            "Weka viungo vya **Google Drive** (kimoja kwa kila mstari; faili lazima liwe *Anyone with the link*) "
            "au viungo vya moja kwa moja vya video. Inapakua, inabadilisha kuwa **MP4 (H.264 + AAC)** inayocheza kila mahali, "
            "na inaweza kuipakia kwenye HF Dataset ili upate **kiungo cha moja kwa moja**."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                m_links = gr.Textbox(label="Viungo (kimoja kwa kila mstari)", lines=5,
                                     placeholder="https://drive.google.com/file/d/....../view")
                m_mode = gr.Radio(choices=["Auto (inapendekezwa)", "Remux tu (haraka, bila kupoteza ubora)", "Re-encode kamili (H.264 + AAC)"],
                                  value="Auto (inapendekezwa)", label="Njia ya kubadilisha")
                m_res = gr.Radio(choices=["Asili", "720p", "480p"], value="Asili",
                                 label="Ubora (unatumika kama re-encode inafanyika)")
                m_up = gr.Checkbox(label="Pakia kwenye HF Dataset (pata kiungo cha moja kwa moja)", value=False)
                m_repo = gr.Textbox(label="Dataset repo", placeholder="username/jaynes-videos", lines=1)
                m_tok = gr.Textbox(label="HF Token (write) — au weka HF_TOKEN kwenye Space Secrets", type="password", lines=1)
                m_btn = gr.Button("⚡ CHUKUA & BADILISHA KUWA MP4", variant="primary")
                m_files = gr.File(label="📥 Download MP4", interactive=False, file_count="multiple")
            with gr.Column(scale=2):
                m_log = gr.Textbox(label="Log (live)", lines=18, max_lines=300, elem_id="log",
                                   placeholder="Bandika viungo kisha bonyeza kitufe. Maendeleo yataonekana hapa papo hapo.")
                m_urls = gr.Textbox(label="🔗 Viungo vya moja kwa moja (kama umepakia)", lines=4,
                                    placeholder="Viungo vya MP4 vitaonekana hapa.")
        m_btn.click(fn=run_pipeline, inputs=[m_links, m_mode, m_res, m_up, m_repo, m_tok],
                    outputs=[m_log, m_files, m_urls])
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue().launch()

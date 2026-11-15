#!/usr/bin/env python3
"""Ngwele Drive -> MP4 -> HF, including posts excluded by category filters.

Never deduplicates different titles using Drive ID alone. Extra source posts are
published to ngwele_extra_posts.json; the listing API must merge that manifest.
"""
import argparse
import copy
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import requests

API = os.getenv('NGWELE_API', 'https://ooqvsqmsnwjqvuzbhczx.supabase.co/functions/v1/ngwele-movie-series')
FEED = os.getenv('NGWELE_FEED', 'https://www.ngweletv.com/feeds/posts/default')
REPO = os.getenv('HF_REPO', 'jaynesmaxtv/movie-file')
FAILED = 'ngwele_failed.json'
MANIFEST = 'ngwele_extra_posts.json'
START = time.time()
BATCH = int(os.getenv('BATCH', '100'))
GROUP = max(1, int(os.getenv('UPLOAD_GROUP', '10')))
MAX_PENDING_BYTES = int(float(os.getenv('MAX_PENDING_GB', '6')) * 1e9)
HARD = START + 330 * 60
STOP_NEW = min(START + int(os.getenv('DEADLINE_MIN', '300')) * 60, HARD - 300)
MAX_TRIES = 5  # Caps backoff, NOT lifetime retries.
COOLDOWN_H = float(os.getenv('COOLDOWN_H', '3'))
MAX_BLOCKED_IN_ROW = 25
TMP = Path('work/ngwele-v2')


def log(*args):
    print(*args, flush=True)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def http_json(url, params=None):
    response = requests.get(url, params=params, timeout=(20, 120),
                            headers={'User-Agent': 'JaynesMaxTV/1.0 (licensed partner)'})
    response.raise_for_status()
    return response.json()


def fetch_api_posts():
    posts, offset, seen = [], 0, set()
    while True:
        data = http_json(API, {'kind': 'all', 'offset': offset, 'limit': 50})
        page = data.get('posts', [])
        if not isinstance(page, list):
            raise ValueError('API posts must be a list')
        if not page:
            break
        page_key = tuple(p.get('slug', p.get('url', '')) for p in page)
        if page_key in seen:
            raise RuntimeError('API pagination repeated a page; stopping to avoid a loop')
        seen.add(page_key)
        posts.extend(page)
        offset += len(page)
        if offset >= int(data.get('total', offset)):
            break
    return posts


def fetch_original_posts():
    entries, start, seen = [], 1, set()
    while True:
        feed = http_json(FEED, {'alt': 'json', 'max-results': 150, 'start-index': start})['feed']
        page = feed.get('entry', [])
        if not page:
            break
        page_key = tuple(e.get('id', {}).get('$t', '') for e in page)
        if page_key in seen:
            raise RuntimeError('Blogger repeated a page')
        seen.add(page_key)
        entries.extend(page)
        start += len(page)
        if len(entries) >= int(feed.get('openSearch$totalResults', {}).get('$t', len(entries))):
            break
        time.sleep(0.6)
    return entries


def url_key(url):
    return urlsplit(url).path.rstrip('/')


def drive_id(url):
    parsed = urlsplit(html.unescape(url))
    if (parsed.hostname or '').lower() not in ('drive.google.com', 'drive.usercontent.google.com'):
        return None
    match = re.search(r'/d/([A-Za-z0-9_-]{10,})', parsed.path)
    value = match.group(1) if match else parse_qs(parsed.query).get('id', [''])[0]
    return value if re.fullmatch(r'[A-Za-z0-9_-]{10,}', value) else None


class SourceHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.images = [], []

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == 'a' and data.get('href'):
            value = drive_id(data['href'])
            if value and value not in self.links:
                self.links.append(value)
        if tag == 'img' and data.get('src', '').startswith('https://'):
            self.images.append(data['src'])


def slugify(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', text).strip('-')


def split_title(raw):
    parts = raw.rsplit('|', 1)
    dj = parts[1].strip() if len(parts) == 2 else ''
    if dj:
        dj = re.sub(r'\s+', ' ', dj)
        dj = 'DJ ' + re.sub(r'^dj\s*', '', dj, flags=re.I).title()
        if dj.lower() in ('dj blaki', 'dj black'):
            dj = 'DJ Black'
    title = re.sub(r'\b(?:FULL\s*MOVIE|MOVIE|20\d{2}|19\d{2})\b', ' ', parts[0], flags=re.I)
    title = re.sub(r'\bPART\s*[AB](?:\s*-\s*[AB])?\b', ' ', title, flags=re.I)
    title = re.sub(r'\b(?:PART\s*)?\d+\s*-\s*\d+\s*FINAL\b', ' ', title, flags=re.I)
    title = re.sub(r'[._]+', ' ', title)
    return re.sub(r'\s+', ' ', title).strip().title(), dj or None


def discover_extras(entries, api_posts):
    """Keep API episode paths/order untouched; only supplement omitted URLs."""
    known_urls = {url_key(p.get('url', '')) for p in api_posts if p.get('url')}
    slugs = {p['slug'] for p in api_posts}
    known_drive = {}
    for post in api_posts:
        for ep in post.get('episodes', []):
            known_drive.setdefault(ep.get('drive_id'), set()).add(post.get('title', ''))
    extras, review = [], []
    for entry in entries:
        url = next((l['href'] for l in entry.get('link', []) if l.get('rel') == 'alternate'), '')
        if not url or url_key(url) in known_urls:
            continue
        raw = entry.get('title', {}).get('$t', '')
        parser = SourceHTML()
        parser.feed(entry.get('content', {}).get('$t', ''))
        if not parser.links:
            continue  # Announcements and posts without downloadable video are not movies.
        labels = {c['term'].upper() for c in entry.get('category', [])}
        # Labelled posts absent from API may have been intentionally deduplicated.
        # Require review rather than guessing or overwriting another title's video.
        shared = {d: sorted(known_drive[d]) for d in parser.links if d in known_drive}
        if shared:
            review.append({'title': raw, 'url': url, 'reason': 'Drive ID also used by another post', 'shared': shared})
            continue
        title, dj = split_title(raw)
        if not title:
            review.append({'title': raw, 'url': url, 'reason': 'Title is empty after cleaning'})
            continue
        # Unlabelled multipart content is not safely classified automatically.
        ambiguous = not labels.intersection({'MOVIE', 'CARTOON', 'SEASON'}) and re.search(r'\b(?:PART\s*)?\d+\s*-\s*\d+\s*FINAL\b', raw, re.I)
        if ambiguous:
            review.append({'title': raw, 'url': url, 'reason': 'Movie or series classification needs review'})
            continue
        slug = 'ngwele-' + slugify(title + (' ' + dj if dj else ''))
        if slug in slugs:
            blogger_id = entry.get('id', {}).get('$t', '').rsplit('-', 1)[-1]
            if not blogger_id.isdigit():
                review.append({'title': raw, 'url': url, 'reason': 'Slug collision without a stable source ID'})
                continue
            slug += '-' + blogger_id
        slugs.add(slug)
        episodes = [{'label': f'Part {n}', 'drive_id': value,
                     'drive_url': f'https://drive.google.com/file/d/{value}/view', 'mp4_url': None}
                    for n, value in enumerate(parser.links, 1)]
        extras.append({'source': 'ngwele', 'title': title, 'dj': dj, 'slug': slug, 'url': url,
                       'kind': 'series' if 'SEASON' in labels else 'movie',
                       'lastmod': entry.get('updated', entry.get('published', {})).get('$t', ''),
                       'poster': parser.images[0] if parser.images else None, 'description': None,
                       'dup_of': None, 'total_declared': len(episodes), 'episodes': episodes})
        known_urls.add(url_key(url))
        for d in parser.links:
            known_drive.setdefault(d, set()).add(title)
    return extras, review


def episode_path(post, n):
    slug = post['slug']
    if not re.fullmatch(r'ngwele-[a-zA-Z0-9_-]+', slug):
        raise ValueError(f'Unsafe slug: {slug}')
    return f'videos/ngwele/{slug}/ep{n:02d}.mp4'


def skip_failed(failed, path, now=None):
    """All failures become eligible again; five tries caps delay at 15 hours."""
    f = failed.get(path)
    if not f:
        return False
    if not isinstance(f, dict):
        f = {'n': int(f), 't': 0}
    if not f.get('t'):
        return False
    elapsed = (time.time() if now is None else now) - float(f['t'])
    delay = COOLDOWN_H * 3600 * min(max(int(f.get('n', 1)), 1), MAX_TRIES)
    return elapsed < delay


def mark_failed(failed, path, error):
    f = failed.get(path, {})
    old = int(f.get('n', 0)) if isinstance(f, dict) else int(f)
    failed[path] = {'n': old + 1, 't': time.time(), 'error': str(error)[:250]}


def build_todo(posts, files, failed, selected=''):
    result, seen = [], set()
    for post in posts:
        if selected and selected.lower() not in (post.get('title', '') + ' ' + post.get('slug', '')).lower():
            continue
        for n, ep in enumerate(post.get('episodes', []), 1):
            path = episode_path(post, n)
            if path in seen or path in files or ep.get('mp4_url') or not ep.get('drive_id') or skip_failed(failed, path):
                continue
            seen.add(path)
            result.append({'path': path, 'id': ep['drive_id'], 'title': f"{post['title']} {ep.get('label', n)}"})
    return result


def manifest_data(extras, files):
    posts = copy.deepcopy(extras)
    for post in posts:
        missing = []
        for n, ep in enumerate(post['episodes'], 1):
            path = episode_path(post, n)
            ep['mp4_url'] = f'https://huggingface.co/datasets/{REPO}/resolve/main/{path}' if path in files else None
            if not ep['mp4_url']:
                missing.append({'label': ep['label'], 'drive_id': ep['drive_id']})
        post.update(available=len(post['episodes']) - len(missing), complete=not missing, unconverted=missing)
    return {'source': 'ngwele', 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'posts': posts}


def hf_retry(fn, *args, **kwargs):
    for i in range(4):
        try:
            return fn(*args, **kwargs)
        except Exception as error:
            if i == 3 or not any(x in str(error) for x in ('429', 'Too Many Requests', '502', '503', '504')):
                raise
            delay = 30 * 2 ** i
            if time.time() + delay >= HARD:
                raise TimeoutError('No time left for Hugging Face retry') from error
            log('HF retry in', delay, 'seconds')
            time.sleep(delay)


class DriveBlocked(Exception):
    pass


def looks_like_video(path):
    if not Path(path).exists() or Path(path).stat().st_size < 100_000:
        return False
    with open(path, 'rb') as f:
        head = f.read(512).lower()
    return b'<html' not in head and b'<!doctype' not in head


def download(value, dst):
    remaining = STOP_NEW - time.time()
    if remaining <= 0:
        raise TimeoutError('Download time limit reached')
    try:
        # Child process bounds gdown redirects/downloads as well as fallback HTTP.
        result = subprocess.run([sys.executable, '-m', 'gdown',
                                 f'https://drive.google.com/uc?id={value}', '-O', str(dst), '-q'],
                                capture_output=True, text=True, timeout=min(900, remaining))
        if result.returncode == 0 and looks_like_video(dst):
            return
    except subprocess.TimeoutExpired:
        log('gdown timed out; trying public Drive download')
    except Exception as error:
        log('gdown failed:', str(error)[:160])
    with requests.get('https://drive.usercontent.google.com/download', stream=True,
                      timeout=(20, 60), params={'id': value, 'export': 'download', 'confirm': 't'}) as response:
        if response.status_code != 200 or 'text/html' in response.headers.get('content-type', ''):
            raise DriveBlocked('Drive quota or non-public file')
        with open(dst, 'wb') as f:
            for chunk in response.iter_content(1 << 20):
                if time.time() >= STOP_NEW:
                    raise TimeoutError('Download time limit reached')
                if shutil.disk_usage(TMP).free < 500_000_000:
                    raise OSError('Disk nearly full')
                f.write(chunk)
    if not looks_like_video(dst):
        raise DriveBlocked('Drive returned HTML or an incomplete file')


def run_ffmpeg(cmd):
    remaining = HARD - time.time() - 180  # Reserve time to flush uploads/state.
    if remaining <= 0:
        raise TimeoutError('No time left for ffmpeg')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=remaining)
    if result.returncode:
        lines = result.stderr.strip().splitlines()
        raise RuntimeError('ffmpeg: ' + (lines[-1][:200] if lines else str(result.returncode)))


def convert(src, dst):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,codec_name,pix_fmt',
                             '-of', 'json', str(src)], capture_output=True, text=True, check=True, timeout=60)
    streams = json.loads(result.stdout).get('streams', [])
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    audio = next((s for s in streams if s.get('codec_type') == 'audio'), None)
    if not video:
        raise RuntimeError('No video stream')
    maps = ['-map', '0:v:0', '-map', '0:a:0?']
    tail = ['-movflags', '+faststart', str(dst)]
    if video.get('codec_name') == 'h264' and video.get('pix_fmt') == 'yuv420p' and audio and audio.get('codec_name') == 'aac':
        try:
            run_ffmpeg(['ffmpeg', '-y', '-v', 'error', '-i', str(src), *maps, '-c', 'copy', *tail])
            return 'remux'
        except RuntimeError as error:
            log('Remux failed, encoding:', error)
    enc = ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96k']
    even = ['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1']
    mkv = str(dst) + '.mkv'
    steps = [
        ('re-encode', [['ffmpeg', '-y', '-v', 'error', '-i', str(src), *maps, *enc, *tail]]),
        ('re-encode genpts', [['ffmpeg', '-y', '-v', 'error', '-fflags', '+genpts', '-i', str(src), *maps, *even, *enc,
                             '-avoid_negative_ts', 'make_zero', '-max_muxing_queue_size', '4096', *tail]]),
        ('via mkv', [['ffmpeg', '-y', '-v', 'error', '-fflags', '+genpts', '-i', str(src), *maps, *even, *enc, '-f', 'matroska', mkv],
                     ['ffmpeg', '-y', '-v', 'error', '-fflags', '+genpts', '-i', mkv, '-c', 'copy', '-avoid_negative_ts', 'make_zero', *tail]])]
    last_error = RuntimeError('Conversion failed')
    for mode, cmds in steps:
        try:
            for cmd in cmds:
                run_ffmpeg(cmd)
            return mode
        except RuntimeError as error:
            last_error = error
            log(mode, 'failed:', error)
            for file in (str(dst), mkv):
                Path(file).unlink(missing_ok=True)
    raise last_error


def load_failed(api):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        path = hf_hub_download(REPO, FAILED, repo_type='dataset', token=api.token, force_download=True)
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError('Failed list is not an object')
        return {k: v if isinstance(v, dict) else {'n': int(v), 't': 0} for k, v in data.items()}
    except EntryNotFoundError:
        return {}


def flush(api, pending, failed, files):
    from huggingface_hub import CommitOperationAdd
    if not pending:
        return 0
    operations = [CommitOperationAdd(path_in_repo=t['path'], path_or_fileobj=str(f)) for t, f in pending]
    # Leave pending files intact on failure; an HF outage is not a Drive failure.
    hf_retry(api.create_commit, repo_id=REPO, operations=operations, repo_type='dataset',
             commit_message=f'ngwele: add {len(pending)} videos')
    count = len(pending)
    for task, file in pending:
        files.add(task['path'])
        failed.pop(task['path'], None)
        file.unlink(missing_ok=True)
    pending.clear()
    return count


def migrate(api, files):
    from huggingface_hub import CommitOperationCopy, CommitOperationDelete
    moves = []
    for file in files:
        m = re.fullmatch(r'videos/(ngwele-.+)-ep(\d+)\.mp4', file)
        if m:
            moves.append((file, f'videos/ngwele/{m[1]}/ep{int(m[2]):02d}.mp4'))
    for offset in range(0, len(moves), 100):
        operations, completed = [], []
        for old, new in moves[offset:offset + 100]:
            if new in files:
                log('Migration needs review; destination already exists:', old)
                continue
            operations.extend([CommitOperationCopy(src_path_in_repo=old, path_in_repo=new), CommitOperationDelete(path_in_repo=old)])
            completed.append((old, new))
        if operations:
            hf_retry(api.create_commit, repo_id=REPO, operations=operations, repo_type='dataset', commit_message='ngwele: migrate old paths')
            for old, new in completed:
                files.discard(old)
                files.add(new)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--dry-run', action='store_true', help='Discover only; no Drive downloads or HF writes')
    args_parser.add_argument('--only', default='', help='Limit conversion by title/slug, e.g. hope')
    args_parser.add_argument('--report', default='ngwele-discovery.json')
    args = args_parser.parse_args()
    posts = fetch_api_posts()
    extras, review = discover_extras(fetch_original_posts(), posts)
    write_json(args.report, {'api_posts': len(posts), 'extra_posts': extras, 'needs_review': review})
    log('API:', len(posts), 'Extra posts:', len(extras), 'Need review:', len(review))
    for p in extras:
        log('EXTRA:', p['title'], '|', p['dj'], '|', p['slug'])
    if args.dry_run:
        return
    token = os.getenv('HF_TOKEN')
    if not token:
        raise RuntimeError('Set HF_TOKEN as a GitHub Actions secret; do not put it in this file')
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = set(hf_retry(api.list_repo_files, REPO, repo_type='dataset'))
    migrate(api, files)
    failed = load_failed(api)
    # Preserve any extra posts from previous runs even if Blogger later removes them.
    if MANIFEST in files:
        from huggingface_hub import hf_hub_download
        previous = json.loads(Path(hf_hub_download(REPO, MANIFEST, repo_type='dataset', token=token, force_download=True)).read_text())
        current_urls = {url_key(p['url']) for p in posts + extras}
        extras.extend(p for p in previous.get('posts', []) if url_key(p.get('url', '')) not in current_urls)
    todo = build_todo(posts + extras, files, failed, args.only)
    log(len(todo), 'eligible episodes, batch=', BATCH)
    shutil.rmtree(TMP, ignore_errors=True)
    (TMP / 'out').mkdir(parents=True)
    pending, pending_bytes, done, blocked = [], 0, 0, 0
    try:
        for seq, task in enumerate(todo, 1):
            if done + len(pending) >= BATCH or time.time() >= STOP_NEW or blocked >= MAX_BLOCKED_IN_ROW:
                break
            if pending and shutil.disk_usage(TMP).free < max(2_000_000_000, pending_bytes):
                done += flush(api, pending, failed, files)
                pending_bytes = 0
            cur = TMP / 'cur'
            shutil.rmtree(cur, ignore_errors=True)
            cur.mkdir()
            src, dst = cur / 'in.bin', cur / 'out.mp4'
            try:
           

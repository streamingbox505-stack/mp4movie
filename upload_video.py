#!/usr/bin/env python3
"""
Upload video kwenye HF Dataset kwa ajili ya editing
Tumia: python upload_video.py
"""

import subprocess, sys

def _install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)

_install()

import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN", "")
REPO     = "jaynesmaxtv/movie-file"

def main():
    token = HF_TOKEN
    if not token:
        token = input("HF Token yako: ").strip()

    api = HfApi(token=token)

    video_path = input("Path ya video (e.g. C:/Videos/myvideo.mp4): ").strip()
    if not os.path.exists(video_path):
        print(f"❌ Faili halipatikani: {video_path}")
        return

    # Tumia jina la faili moja kwa moja
    filename  = os.path.basename(video_path)
    repo_path = f"videos/edit/{filename}"

    size_mb = os.path.getsize(video_path) / 1e6
    print(f"\n📤 Itapakiwa: {filename} ({size_mb:.1f} MB)")

    confirm = input("Endelea? (y/n): ").strip().lower()
    if confirm != "y":
        print("Imesimamishwa.")
        return

    print("\n⏳ Inapakia...")
    api.upload_file(
        path_or_fileobj=video_path,
        path_in_repo=repo_path,
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"add {filename}",
    )
    print(f"\n✅ Imepakiwa!")
    print(f"🔗 https://huggingface.co/datasets/{REPO}/blob/main/{repo_path}")

if __name__ == "__main__":
    main()

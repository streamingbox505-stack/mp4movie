#!/usr/bin/env python3
"""compare_subs.py — pima usahihi wa Whisper dhidi ya manukuu ya kweli (rasmi), na uone mistari mibaya zaidi.

Matumizi:
    python compare_subs.py work/segments.json subs/jina.srt        # segments.json huandikwa na dub_movie.py
Matokeo: WER (asilimia ya maneno yenye makosa), idadi ya mistari iliyopatikana, na mistari 15 mibaya zaidi (kweli | Whisper).
Hifadhi: work/quality.json na work/quality.txt
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dub_movie import estimate_shift, parse_subs


def words(t):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", t.lower())


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def compare(segs, cues):
    off, d, rate = estimate_shift([s["start"] for s in segs], [c[0] for c in cues])
    errs = refw = found = 0
    rows = []
    for cs, ce, text in cues:
        cs, ce = cs * (1 + d) + off, ce * (1 + d) + off
        hit = [s for s in segs if s["start"] < ce - 0.1 and s["end"] > cs + 0.1]
        ref, hyp = words(text), words(" ".join(s["text"] for s in hit))
        if not ref:
            continue
        found += bool(hit)
        e = edit_distance(ref, hyp) if hit else len(ref)
        errs += e; refw += len(ref)
        rows.append((e / len(ref), e, round(cs, 1), text, " ".join(s["text"] for s in hit)))
    rows.sort(key=lambda r: (-r[1], -r[0]))
    return {"offset_s": round(off, 2), "drift_pct": round(d * 100, 3), "match_rate": round(rate, 2), "cues": len(cues), "cues_with_speech": found,
            "wer_pct": round(100 * errs / max(1, refw), 1), "ref_words": refw}, rows


def main():
    segp, subp = (sys.argv + [None, None])[1:3]
    if not segp or not subp:
        sys.exit(__doc__)
    segs = json.load(open(segp, encoding="utf-8"))
    segs = segs["segments"] if isinstance(segs, dict) else segs
    cues = parse_subs(open(subp, encoding="utf-8-sig", errors="replace").read())
    rep, rows = compare(segs, cues)
    lines = [f"WER: {rep['wer_pct']}%  (makosa ya maneno dhidi ya manukuu ya kweli; chini = bora)",
             f"Mistari ya manukuu: {rep['cues']} | yenye usemi uliotambuliwa: {rep['cues_with_speech']} | offset {rep['offset_s']}s, drift {rep['drift_pct']}%", "",
             "Mistari 15 mibaya zaidi (dakika | kweli | Whisper):"]
    lines += [f"{r[2] / 60:5.1f} dk | {r[3][:90]} | {r[4][:90] or '(hakuna)'}" for r in rows[:15]]
    print("\n".join(lines))
    os.makedirs(os.path.dirname(segp) or ".", exist_ok=True)
    json.dump({**rep, "worst": [{"min": round(r[2] / 60, 2), "ref": r[3], "whisper": r[4], "errors": r[1]} for r in rows[:50]]},
              open(os.path.join(os.path.dirname(segp) or ".", "quality.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(os.path.dirname(segp) or ".", "quality.txt"), "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()

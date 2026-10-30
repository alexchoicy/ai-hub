"""Experiment: ASR-guided chunked forced alignment of song lyrics.

usage: chunk_align.py LYRICS OUT_DIR NAME=AUDIO [NAME=AUDIO ...]
Prints metrics only (line indices, no lyric text). Writes qwen_NAME.json + qwen_NAME.lrc.
"""

import itertools
import json
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import pykakasi
import torch
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

SR = 16000
CHUNK, OVERLAP = 30.0, 5.0
MIN_SCORE = 0.35

kks = pykakasi.kakasi()


def reading(s: str) -> str:
    """Hiragana reading with only letters/digits, so kanji and kana spellings match."""
    hira = "".join(p["hira"] for p in kks.convert(s))
    return "".join(c for c in hira if unicodedata.category(c)[0] in "LN")


def match_span(asr: str, readings: list[str], ptr: int) -> tuple[int, int, float]:
    best = (0, 0, 0.0)
    for i in range(max(0, ptr - 3), min(len(readings), ptr + 10)):
        for j in range(i + 1, min(len(readings), i + 10) + 1):
            score = SequenceMatcher(None, asr, "".join(readings[i:j])).ratio()
            if score > best[2]:
                best = (i, j, score)
    return best


def run(name, path, lines, readings, asr_model, aligner, out_dir):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(audio) / SR
    starts = np.arange(0, max(dur - OVERLAP, 1e-6), CHUNK - OVERLAP)
    chunks = [(float(s), float(min(s + CHUNK, dur))) for s in starts]

    t0 = time.time()
    asr = asr_model.transcribe(
        audio=[(audio[int(s * SR) : int(e * SR)], SR) for s, e in chunks],
        language="Japanese",
    )
    t_asr = time.time() - t0

    ptr = 0
    cands: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    chunk_log = []
    t0 = time.time()
    for (s, e), tr in zip(chunks, asr):
        asr_read = reading(tr.text)
        if len(asr_read) < 3:
            chunk_log.append({"chunk": [s, e], "skip": "no speech"})
            continue
        i, j, score = match_span(asr_read, readings, ptr)
        if score < MIN_SCORE:
            chunk_log.append({"chunk": [s, e], "skip": f"score {score:.2f}"})
            continue
        ptr = max(ptr, j - 1)
        text = " ".join(lines[i:j])
        [res] = aligner.align(
            audio=(audio[int(s * SR) : int(e * SR)], SR), text=text, language="Japanese"
        )
        per_line = [
            len(aligner.aligner_processor.encode_timestamp(l, "japanese")[0])
            for l in lines[i:j]
        ]
        keys = [(li, k) for li, cnt in zip(range(i, j), per_line) for k in range(cnt)]
        if len(keys) != len(res.items):
            chunk_log.append({"chunk": [s, e], "skip": "token mismatch"})
            continue
        lo = s if s > 0 else -1e9
        hi = e if e < dur else 1e9
        for key, it in zip(keys, res.items):
            a, b = s + it.start_time, s + it.end_time
            mid = (a + b) / 2
            cands.setdefault(key, []).append((min(mid - lo, hi - mid), a, b))
        chunk_log.append({"chunk": [s, e], "lines": [i, j], "score": round(score, 2)})
    t_align = time.time() - t0

    words = {k: max(v)[1:] for k, v in cands.items()}
    line_times = {}
    for (li, _), (a, b) in sorted(words.items()):
        lt = line_times.setdefault(li, [a, b])
        lt[0], lt[1] = min(lt[0], a), max(lt[1], b)

    keys = sorted(words)
    zero = sum(1 for k in keys if words[k][1] - words[k][0] <= 1e-6)
    backwards = sum(
        1 for a, b in itertools.pairwise(keys) if words[b][0] < words[a][0] - 1e-6
    )
    long_words = sum(1 for k in keys if words[k][1] - words[k][0] > 3.0)
    scores = [c["score"] for c in chunk_log if "score" in c]

    out = {
        "audio": str(path),
        "duration": dur,
        "chunks": chunk_log,
        "lines": {
            str(li): {"start": round(a, 2), "end": round(b, 2)}
            for li, (a, b) in sorted(line_times.items())
        },
        "words": [
            {
                "line": li,
                "k": k,
                "start": round(words[(li, k)][0], 3),
                "end": round(words[(li, k)][1], 3),
            }
            for li, k in keys
        ],
    }
    Path(out_dir, f"qwen_{name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1)
    )
    with open(Path(out_dir, f"qwen_{name}.lrc"), "w") as f:
        for li, (a, _) in sorted(line_times.items()):
            f.write(f"[{int(a // 60):02d}:{a % 60:05.2f}]{lines[li]}\n")

    print(f"\n== {name}  (asr {t_asr:.1f}s, align {t_align:.1f}s)")
    for c in chunk_log:
        tag = (
            f"lines {c['lines'][0]:>2}-{c['lines'][1] - 1:<2} score {c['score']:.2f}"
            if "lines" in c
            else f"SKIP ({c['skip']})"
        )
        print(f"  {c['chunk'][0]:6.1f}-{c['chunk'][1]:6.1f}s  {tag}")
    print(f"  lines placed: {len(line_times)}/{len(lines)}   words: {len(keys)}")
    print(
        f"  zero-length words: {zero} ({zero / max(len(keys), 1):.0%})   >3s words: {long_words}   out-of-order: {backwards}"
    )
    print(f"  mean match score: {np.mean(scores) if scores else 0:.2f}")
    return {li: a for li, (a, _) in line_times.items()}


def main():
    lyrics, out_dir, *specs = sys.argv[1:]
    lines = [l.strip() for l in Path(lyrics).read_text().splitlines() if l.strip()]
    readings = [reading(l) for l in lines]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    aligner = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda:0"
    )
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=256,
    )

    starts = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        starts[name] = run(name, path, lines, readings, asr_model, aligner, out_dir)

    names = list(starts)
    print("\n== agreement of line start times (median |diff|, lines in both)")
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = starts[names[x]], starts[names[y]]
            common = sorted(set(a) & set(b))
            diffs = [abs(a[l] - b[l]) for l in common]
            print(
                f"  {names[x]} vs {names[y]}: {np.median(diffs) if diffs else float('nan'):.2f}s over {len(common)} lines, >2s apart: {sum(d > 2 for d in diffs)}"
            )


if __name__ == "__main__":
    main()

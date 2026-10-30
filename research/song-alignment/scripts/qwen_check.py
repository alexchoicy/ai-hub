"""Cross-check: Qwen3-ForcedAligner (Cantonese) on the same chunk spans; compare line starts."""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from qwen_asr import Qwen3ForcedAligner

lyrics, audio_path, ctc_json = sys.argv[1:]
lines = [l.strip() for l in Path(lyrics).read_text().splitlines() if l.strip()]
d = json.loads(Path(ctc_json).read_text())
ctc_start = {}
for w in d["words"]:
    ctc_start.setdefault(w["line"], w["start"])
audio, _ = librosa.load(audio_path, sr=16000, mono=True)
al = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda:0"
)
for c in d["chunks"]:
    if "lines" not in c:
        continue
    s, e = c["chunk"]
    i, j = c["lines"]
    [r] = al.align(
        audio=(audio[int(s * 16000) : int(e * 16000)], 16000),
        text=" ".join(lines[i:j]),
        language="Cantonese",
    )
    zero = sum(it.end_time <= it.start_time for it in r.items) / max(len(r.items), 1)
    # map tokens back to lines by per-line token counts
    counts = [len(al.aligner_processor.tokenize_space_lang(l)) for l in lines[i:j]]
    diffs, pos = [], 0
    for li, n in zip(range(i, j), counts):
        if n and pos < len(r.items) and li in ctc_start:
            q = s + r.items[pos].start_time
            if s + 3 < ctc_start[li] < e - 3:  # skip lines near chunk edges
                diffs.append(abs(q - ctc_start[li]))
        pos += n
    ok = "ok " if zero < 0.2 else "BAD"
    print(
        f"  {s:5.0f}s qwen zero {zero:4.0%} {ok}  lines compared {len(diffs):2d}  median |qwen-ctc| {np.median(diffs) if diffs else float('nan'):5.2f}s  >1s: {sum(x > 1 for x in diffs)}"
    )

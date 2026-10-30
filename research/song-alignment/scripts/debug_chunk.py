import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from qwen_asr import Qwen3ForcedAligner

lyrics, path, s, e, i, j = (
    sys.argv[1],
    sys.argv[2],
    float(sys.argv[3]),
    float(sys.argv[4]),
    int(sys.argv[5]),
    int(sys.argv[6]),
)
lines = [l.strip() for l in Path(lyrics).read_text().splitlines() if l.strip()]
audio, _ = librosa.load(path, sr=16000, mono=True)
clip = audio[int(s * 16000) : int(e * 16000)]
al = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda:0"
)
print("clip peak", float(np.abs(clip).max()), "rms", float(np.sqrt((clip**2).mean())))

# patch to see raw timestamps before fix_timestamp
proc = al.aligner_processor
orig = proc.fix_timestamp


def spy(ts):
    print("  raw ms:", [int(x) for x in ts[:24]], "...", "n=", len(ts))
    return orig(ts)


proc.fix_timestamp = spy

for label, text in [
    ("space-joined", " ".join(lines[i:j])),
    ("newline-joined", "\n".join(lines[i:j])),
]:
    for dt in ["bf16", "fp32"]:
        if dt == "fp32":
            al.model.to(torch.float32)
        else:
            al.model.to(torch.bfloat16)
        [r] = al.align(audio=(clip, 16000), text=text, language="Japanese")
        z = sum(1 for it in r.items if it.end_time - it.start_time <= 0)
        print(
            f"{label} {dt}: words={len(r.items)} zero={z} span={r.items[0].start_time}-{r.items[-1].end_time}"
        )

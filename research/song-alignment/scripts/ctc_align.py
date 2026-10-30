"""Experiment: MMS CTC forced alignment per 30 s chunk, reusing chunk->lines from chunk_align.py.

usage: ctc_align.py LYRICS OUT_DIR NAME=AUDIO [NAME=AUDIO ...]
Reads OUT_DIR/qwen_NAME.json (chunk spans), writes OUT_DIR/ctc_NAME.json + ctc_NAME.lrc.
Prints metrics only.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pykakasi
import torch
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
from transformers import AutoTokenizer, Wav2Vec2ForCTC

SR = 16000
MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
kks = pykakasi.kakasi()


_ja = Qwen3ForceAlignProcessor()


def line_words(line: str, vocab: dict[str, int]) -> list[list[int]]:
    """Split a lyric line into words (nagisa, same as Qwen's aligner), then
    romanize each word into char token ids."""
    words = []
    for w in _ja.tokenize_japanese(line):
        roman = "".join(p["hepburn"] for p in kks.convert(w)).lower()
        ids = [vocab[c] for c in roman if c in vocab and c not in "<>"]
        if ids:
            words.append(ids)
    return words


def viterbi(
    logp: np.ndarray, tokens: list[int], blank: int = 0
) -> tuple[np.ndarray, float]:
    """Best CTC path. Returns the target-token index active at each frame (-1 = blank)."""
    ext = [blank]
    for t in tokens:
        ext += [t, blank]
    ext = np.array(ext)
    T, S = logp.shape[0], len(ext)
    emit = logp[:, ext]  # (T, S)
    skip_ok = np.zeros(S, bool)
    skip_ok[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    dp = np.full(S, -np.inf)
    dp[0], dp[1] = emit[0, 0], emit[0, 1]
    bp = np.zeros((T, S), np.int8)
    for t in range(1, T):
        stay = dp
        one = np.concatenate([[-np.inf], dp[:-1]])
        two = np.where(skip_ok, np.concatenate([[-np.inf, -np.inf], dp[:-2]]), -np.inf)
        cand = np.stack([stay, one, two])
        bp[t] = cand.argmax(0)
        dp = cand.max(0) + emit[t]

    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    score = float(dp[s])
    states = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s = int(s) - int(bp[t, s])
    token_idx = np.where(ext[states] == blank, -1, (states - 1) // 2)
    return token_idx, score


def run(name, path, lines, model, vocab, out_dir):
    meta = json.loads(Path(out_dir, f"qwen_{name}.json").read_text())
    audio, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(audio) / SR

    cands = {}
    log = []
    t0 = time.time()
    for c in meta["chunks"]:
        if "lines" not in c:
            continue
        s, e = c["chunk"]
        i, j = c["lines"]
        clip = audio[int(s * SR) : int(e * SR)]
        x = (
            torch.from_numpy((clip - clip.mean()) / (clip.std() + 1e-7))[None]
            .cuda()
            .half()
        )
        with torch.inference_mode():
            logp = torch.log_softmax(model(x).logits[0].float(), -1).cpu().numpy()
        frame = len(clip) / SR / logp.shape[0]

        keys, tokens, owner = [], [], []
        for li in range(i, j):
            for k, ids in enumerate(line_words(lines[li], vocab)):
                keys.append((li, k))
                owner += [len(keys) - 1] * len(ids)
                tokens += ids
        if len(tokens) >= logp.shape[0]:
            log.append({"chunk": [s, e], "skip": "text longer than audio"})
            continue
        tok_at, _score = viterbi(logp, tokens)

        owner = np.array(owner)
        spans = {}
        confs = []
        for f, ti in enumerate(tok_at):
            if ti < 0:
                continue
            w = owner[ti]
            a, b = spans.get(w, (f, f))
            spans[w] = (min(a, f), max(b, f + 1))
            confs.append(logp[f, tokens[ti]])
        conf = float(np.exp(np.mean(confs)))

        lo = s if s > 0 else -1e9
        hi = e if e < dur else 1e9
        for w, (fa, fb) in spans.items():
            a, b = s + fa * frame, s + fb * frame
            mid = (a + b) / 2
            cands.setdefault(keys[w], []).append((min(mid - lo, hi - mid), a, b))
        log.append({"chunk": [s, e], "lines": [i, j], "conf": round(conf, 3)})
    t_align = time.time() - t0

    words = {k: max(v)[1:] for k, v in cands.items()}
    line_times = {}
    for (li, _), (a, b) in sorted(words.items()):
        lt = line_times.setdefault(li, [a, b])
        lt[0], lt[1] = min(lt[0], a), max(lt[1], b)
    keys = sorted(words)
    backwards = sum(
        1 for a, b in itertools.pairwise(keys) if words[b][0] < words[a][0] - 1e-6
    )
    long_words = sum(1 for k in keys if words[k][1] - words[k][0] > 3.0)

    Path(out_dir, f"ctc_{name}.json").write_text(
        json.dumps(
            {
                "chunks": log,
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
            },
            indent=1,
        )
    )
    with open(Path(out_dir, f"ctc_{name}.lrc"), "w") as f:
        for li, (a, _) in sorted(line_times.items()):
            f.write(f"[{int(a // 60):02d}:{a % 60:05.2f}]{lines[li]}\n")

    print(f"\n== {name} (ctc align {t_align:.1f}s)")
    print(
        "  chunk confidence: "
        + "  ".join(
            f"{int(c['chunk'][0])}s:{c['conf']:.2f}" for c in log if "conf" in c
        )
    )
    print(
        f"  lines placed: {len(line_times)}/{len(lines)}  words: {len(keys)}  >3s words: {long_words}  out-of-order: {backwards}"
    )
    return {li: a for li, (a, _) in line_times.items()}


def main():
    lyrics, out_dir, *specs = sys.argv[1:]
    lines = [l.strip() for l in Path(lyrics).read_text().splitlines() if l.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL)
    vocab = tok.get_vocab()
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

    starts = {
        n: run(n, p, lines, model, vocab, out_dir)
        for n, p in (s.split("=", 1) for s in specs)
    }

    qwen = {}
    for n in starts:
        d = json.loads(Path(out_dir, f"qwen_{n}.json").read_text())
        good = {}
        for c in d["chunks"]:
            if "lines" not in c:
                continue
            s, e = c["chunk"]
            i, j = c["lines"]
            ws = [
                w
                for w in d["words"]
                if i <= w["line"] < j and s - 0.01 <= w["start"] <= e + 0.01
            ]
            if ws and sum(w["end"] - w["start"] <= 1e-6 for w in ws) / len(ws) < 0.2:
                for li in range(i, j):
                    if str(li) in d["lines"]:
                        good[li] = d["lines"][str(li)]["start"]
        qwen[n] = good

    print("\n== CTC line starts vs Qwen on chunks where Qwen worked (<20% zero-length)")
    for n, st in starts.items():
        common = sorted(set(st) & set(qwen[n]))
        diffs = [abs(st[l] - qwen[n][l]) for l in common]
        print(
            f"  {n}: {len(common)} lines, median |diff| {np.median(diffs) if diffs else float('nan'):.2f}s, >1s apart: {sum(d > 1 for d in diffs)}"
        )

    names = list(starts)
    print("\n== CTC agreement between versions (median |diff| of line starts)")
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = starts[names[x]], starts[names[y]]
            common = sorted(set(a) & set(b))
            diffs = [abs(a[l] - b[l]) for l in common]
            print(
                f"  {names[x]} vs {names[y]}: {np.median(diffs):.2f}s over {len(common)} lines, >1s apart: {sum(d > 1 for d in diffs)}"
            )


if __name__ == "__main__":
    main()

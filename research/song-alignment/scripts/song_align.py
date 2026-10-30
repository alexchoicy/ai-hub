"""Experiment: song lyrics alignment = 30 s chunks -> Qwen3-ASR (locate lines) -> MMS CTC (align).

usage: song_align.py LANG(ja|yue) LYRICS AUDIO OUT_PREFIX
Prints metrics only. Writes OUT_PREFIX.json and OUT_PREFIX.lrc.
"""

import itertools
import json
import os
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from transformers import AutoTokenizer, Wav2Vec2ForCTC

SR = 16000
CHUNK, OVERLAP = 30.0, 5.0
MIN_SCORE = 0.35
CTC_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"


# ---------- language-specific text handling ----------


class Japanese:
    asr_language = "Japanese"

    def __init__(self):
        import pykakasi
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor

        self.kks = pykakasi.kakasi()
        self.proc = Qwen3ForceAlignProcessor()

    def words(self, line):
        return self.proc.tokenize_japanese(line)

    def roman(self, word):
        return "".join(p["hepburn"] for p in self.kks.convert(word))

    def match_key(self, text):
        hira = "".join(p["hira"] for p in self.kks.convert(text))
        return "".join(c for c in hira if unicodedata.category(c)[0] in "LN")


class Cantonese:
    asr_language = "Cantonese"

    def __init__(self):
        import opencc
        import pycantonese

        self.t2 = opencc.OpenCC("s2hk")
        self.pc = pycantonese

    def _jyut(self, text):
        out = []
        for chars, jp in self.pc.characters_to_jyutping(self.t2.convert(text)):
            if jp is None:
                out.append(chars)
            else:
                out.extend(re.sub(r"\d", "", jp).split())
        return out

    def words(self, line):
        # one unit per CJK character, latin runs split on non-letters
        return re.findall(r"[㐀-鿿豈-﫿]|[A-Za-z']+", line)

    def roman(self, word):
        if re.fullmatch(r"[A-Za-z']+", word):
            return word
        if os.environ.get("ROMAN") == "pinyin":
            from pypinyin import lazy_pinyin

            return "".join(lazy_pinyin(word))
        return "".join(self._jyut(word))

    def match_key(self, text):
        # syllables as single symbols, so homophones/simplified chars still match
        return [s.lower() for s in self._jyut(text) if re.search(r"[a-z]", s.lower())]


# ---------- CTC ----------


def viterbi(logp, tokens, blank=0):
    ext = [blank]
    for t in tokens:
        ext += [t, blank]
    ext = np.array(ext)
    T, S = logp.shape[0], len(ext)
    emit = logp[:, ext]
    skip_ok = np.zeros(S, bool)
    skip_ok[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    dp = np.full(S, -np.inf)
    dp[0], dp[1] = emit[0, 0], emit[0, 1]
    bp = np.zeros((T, S), np.int8)
    for t in range(1, T):
        one = np.concatenate([[-np.inf], dp[:-1]])
        two = np.where(skip_ok, np.concatenate([[-np.inf, -np.inf], dp[:-2]]), -np.inf)
        cand = np.stack([dp, one, two])
        bp[t] = cand.argmax(0)
        dp = cand.max(0) + emit[t]
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    states = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s = int(s) - int(bp[t, s])
    return np.where(ext[states] == blank, -1, (states - 1) // 2)


def match_span(asr_key, keys, ptr):
    best = (0, 0, 0.0)
    for i in range(max(0, ptr - 3), min(len(keys), ptr + 10)):
        for j in range(i + 1, min(len(keys), i + 10) + 1):
            joined = [x for k in keys[i:j] for x in k]
            score = SequenceMatcher(None, asr_key, joined, autojunk=False).ratio()
            if score > best[2]:
                best = (i, j, score)
    return best


def main():
    lang, lyrics, audio_path, prefix = sys.argv[1:]
    L = {"ja": Japanese, "yue": Cantonese}[lang]()
    lines = [l.strip() for l in Path(lyrics).read_text().splitlines() if l.strip()]
    line_keys = [L.match_key(l) for l in lines]

    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    dur = len(audio) / SR
    chunks = [
        (float(s), float(min(s + CHUNK, dur)))
        for s in np.arange(0, max(dur - OVERLAP, 1e-6), CHUNK - OVERLAP)
    ]

    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=256,
    )
    t0 = time.time()
    asr = asr_model.transcribe(
        audio=[(audio[int(s * SR) : int(e * SR)], SR) for s, e in chunks],
        language=L.asr_language,
    )
    t_asr = time.time() - t0
    del asr_model
    torch.cuda.empty_cache()

    # 1) locate lines per chunk
    spans, ptr = [], 0
    for (s, e), tr in zip(chunks, asr):
        k = L.match_key(tr.text)
        if len(k) < 3:
            spans.append(None)
            continue
        i, j, score = match_span(k, line_keys, ptr)
        if score < MIN_SCORE:
            spans.append(None)
            continue
        ptr = max(ptr, j - 1)
        spans.append([i, j, score])
    # fill lines skipped between consecutive matched chunks into the later chunk
    prev_end = 0
    for sp in spans:
        if sp is None:
            continue
        sp[0] = min(sp[0], prev_end)
        prev_end = max(prev_end, sp[1])

    # 2) CTC align each chunk with its lines
    tok = AutoTokenizer.from_pretrained(CTC_MODEL)
    vocab = tok.get_vocab()
    model = Wav2Vec2ForCTC.from_pretrained(CTC_MODEL, dtype=torch.float16).cuda().eval()
    letters = {c: i for c, i in vocab.items() if len(c) == 1}

    cands, log = {}, []
    t0 = time.time()
    for (s, e), sp in zip(chunks, spans):
        if sp is None:
            log.append({"chunk": [s, e], "skip": True})
            continue
        i, j, score = sp
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
            for k, w in enumerate(L.words(lines[li])):
                ids = [letters[c] for c in L.roman(w).lower() if c in letters]
                if not ids:
                    continue
                keys.append((li, k, w))
                owner += [len(keys) - 1] * len(ids)
                tokens += ids
        if len(tokens) >= logp.shape[0]:
            log.append({"chunk": [s, e], "skip": True})
            continue
        tok_at = viterbi(logp, tokens)
        owner = np.array(owner)
        wspan, confs = {}, []
        for f, ti in enumerate(tok_at):
            if ti < 0:
                continue
            w = owner[ti]
            a, b = wspan.get(w, (f, f))
            wspan[w] = (min(a, f), max(b, f + 1))
            confs.append(logp[f, tokens[ti]])
        lo = s if s > 0 else -1e9
        hi = e if e < dur else 1e9
        for w, (fa, fb) in wspan.items():
            a, b = s + fa * frame, s + fb * frame
            mid = (a + b) / 2
            cands.setdefault(keys[w], []).append((min(mid - lo, hi - mid), a, b))
        log.append(
            {
                "chunk": [s, e],
                "lines": [i, j],
                "score": round(score, 2),
                "conf": round(float(np.exp(np.mean(confs))), 2),
            }
        )
    t_ctc = time.time() - t0

    words = {k: max(v)[1:] for k, v in cands.items()}
    keys = sorted(words, key=lambda k: (k[0], k[1]))
    line_times = {}
    for (li, _, _), (a, b) in ((k, words[k]) for k in keys):
        lt = line_times.setdefault(li, [a, b])
        lt[0], lt[1] = min(lt[0], a), max(lt[1], b)

    Path(prefix + ".json").write_text(
        json.dumps(
            {
                "duration": dur,
                "chunks": log,
                "words": [
                    {
                        "line": li,
                        "text": w,
                        "start": round(words[(li, k, w)][0], 3),
                        "end": round(words[(li, k, w)][1], 3),
                    }
                    for li, k, w in keys
                ],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    with open(prefix + ".lrc", "w") as f:
        for li, (a, _) in sorted(line_times.items()):
            f.write(f"[{int(a // 60):02d}:{a % 60:05.2f}]{lines[li]}\n")

    durs = [words[k][1] - words[k][0] for k in keys]
    backwards = sum(
        1 for a, b in itertools.pairwise(keys) if words[b][0] < words[a][0] - 1e-6
    )
    lt = sorted(line_times.items())
    gaps = [
        (a, round(line_times[b][0] - line_times[a][1], 1))
        for (a, _), (b, _) in itertools.pairwise(lt)
    ]
    print(f"duration {dur:.1f}s  asr {t_asr:.1f}s  ctc {t_ctc:.1f}s")
    for c in log:
        tag = (
            f"lines {c['lines'][0]:>2}-{c['lines'][1] - 1:<2} match {c['score']:.2f} conf {c['conf']:.2f}"
            if "lines" in c
            else "SKIP"
        )
        print(f"  {c['chunk'][0]:6.1f}-{c['chunk'][1]:6.1f}s  {tag}")
    print(
        f"lines placed {len(line_times)}/{len(lines)}  words {len(keys)}  out-of-order {backwards}"
    )
    print(
        f"word dur median {np.median(durs):.2f}  p90 {np.percentile(durs, 90):.2f}  max {max(durs):.2f}  zero {sum(d <= 0 for d in durs)}"
    )
    print("line starts:", [round(a, 1) for _, (a, _) in lt])
    print("gaps >4s after line:", [g for g in gaps if g[1] > 4])
    print("missing lines:", [i for i in range(len(lines)) if i not in line_times])


if __name__ == "__main__":
    main()

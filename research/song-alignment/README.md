# Song lyrics alignment

Can we get per-word timestamps for sung lyrics, given the audio and the lyrics?
Qwen3-ForcedAligner alone is unreliable on music, so this folder tests a pipeline
around it.

**Result:** Mel-Band RoFormer vocals → 30 s chunks (5 s overlap) → Qwen3-ASR to
find which lyric lines each chunk contains → MMS CTC forced alignment → merge.
Works on a Japanese and a Cantonese song, with no zero-length words.

## Layout

```
scripts/
  separate.sh     vocal stem with Mel-Band / BS-RoFormer (audio-separator via uvx)
  song_align.py   the final pipeline: ASR line matching + MMS CTC, `ja` or `yue`
  chunk_align.py  earlier: ASR line matching + Qwen3-ForcedAligner per chunk
  ctc_align.py    earlier: MMS CTC on the chunk spans found by chunk_align.py
  qwen_check.py   cross-check: Qwen3-ForcedAligner on song_align.py's chunks
  debug_chunk.py  dump Qwen3-ForcedAligner's raw timestamps for one chunk
data/             audio, lyrics and outputs (git-ignored: large and copyrighted)
  ishu/           GESO & rinri - 異種 (Japanese, 242 s, Ogg Opus)
  kasukabe/       林家謙 - 春日部 (Cantonese, 182 s, FLAC 96 kHz)
```

Each song folder has `song.*`, `lyrics.txt` (one sung line per line, no
translations), `vocals_*.wav` and `out/*.json|lrc`. The `.lrc` files can be
loaded in a music player to check the timing by ear.

## Running

```sh
uv sync --group research
scripts/separate.sh data/kasukabe/song.flac data/kasukabe/vocals_melband.wav
ROMAN=pinyin uv run --group research python -I scripts/song_align.py \
  yue data/kasukabe/lyrics.txt data/kasukabe/vocals_melband.wav data/kasukabe/out/ctc_pinyin
```

Paths are relative to this folder. `song_align.py` takes `ja` (Hepburn romaji
for CTC) or `yue` (Jyutping by default, `ROMAN=pinyin` for pinyin). It prints
metrics only and writes `OUT_PREFIX.json` (per-word `line`, `text`, `start`,
`end`) and `OUT_PREFIX.lrc`.

## Findings

### 1. Qwen3-ForcedAligner on whole songs fails

- It rejects audio over 180 s, so a full song can't be aligned in one call.
- On the first 170 s of 異種 (original mix) it ran, but 60/167 words had zero
  length and the rest were bunched or spread at suspiciously even ~3.9 s steps.

### 2. Chunking needs to know which lines are in each chunk

A forced aligner squeezes all the given text into the given audio, so each
30 s chunk must get only the lines sung in it. Qwen3-ASR-1.7B transcribes each
chunk; the transcript is matched against the lyrics (monotonic, whole lines
only) in a phonetic form so spelling differences don't matter:

- Japanese: hiragana reading (pykakasi), so kanji and kana verses match.
- Cantonese: toneless Jyutping syllables (pycantonese), after converting ASR
  output to traditional characters (OpenCC `s2hk`); simplified characters have
  no Jyutping entry.

Match scores were 0.72–0.99 (異種) and 0.58–0.89 (春日部). Chunks with no
vocals are skipped. Lines skipped between two matched chunks are added to the
later chunk (fixed a missing line in 異種).

Overlapping words are taken from the chunk where they are furthest from the
chunk edge.

### 3. Vocal separation helps; Mel-Band is the better choice

異種, chunked pipeline with Qwen3-ForcedAligner:

| | Original mix | BS-RoFormer (Viperx-1297) | Mel-Band RoFormer (Kim) |
|---|---|---|---|
| Separation time | – | 2 min 7 s | 17 s |
| Lyrics match score | 0.87 | 0.90 | 0.91 |
| Zero-length words | 97% | 75% | 66% |

### 4. Qwen3-ForcedAligner fails per chunk, all or nothing

Some chunks aligned well (Mel-Band 50 s: 2/17 zero-length), others failed in
every version (125 s, 175 s, 200 s: all zero). On failing chunks the model's
raw timestamps are noise, and qwen-asr's `fix_timestamp` then collapses them:
it keeps the longest non-decreasing run (`<=`), which a run of zeros wins.
Speech-trained, it does not cope with singing in Japanese. It did better on
Cantonese (8–47% zero-length per chunk).

### 5. MMS CTC alignment works

`MahmoudAshraf/mms-300m-1130-forced-aligner` (same weights as torchaudio's
MMS_FA, loaded with transformers; Viterbi implemented in `song_align.py`).
CTC always yields ordered timestamps of at least one 20 ms frame, so it cannot
collapse.

Words are split like Qwen's aligner (nagisa for Japanese, one unit per
character for Chinese, latin runs as words), each romanized separately; a
word's span is its first to last letter frame.

| | 異種 (Mel-Band) | 春日部 (Mel-Band, pinyin) |
|---|---|---|
| Lines placed | 48/49 (before gap fill) | 40/40 |
| Words | 230 | 346 |
| Zero-length words | 0 | 0 |
| Word duration median / p90 / max | 0.22 / 0.80 / 2.00 s | 0.14 / 0.24 / 3.64 s |
| Out-of-order words | 1 | 1 |
| Alignment time, whole song | 1.6 s | 0.8 s |

Checks (no hand-labelled ground truth yet):

- 異種: on chunks where Qwen3-ForcedAligner worked, line starts agree with CTC
  to ~0.02 s. Instrumental breaks show up as real gaps (26 s, 23 s). The three
  versions (mix, BS, Mel-Band) agree: median line-start difference 0.00 s.
- 春日部: Qwen3-ForcedAligner agrees with CTC to 0.02–0.05 s on chunks 0–100 s;
  0.4 s apart at 125–150 s where both are less sure.

### 6. Romanization for CTC: pinyin, even for Cantonese

MMS was trained on uroman-romanized text, which writes Chinese characters as
Mandarin pinyin. On 春日部, mean CTC confidence per chunk:

| Chunk start | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|---|
| Jyutping | 0.02 | 0.04 | 0.06 | 0.02 | 0.05 | 0.07 | 0.03 |
| Pinyin | 0.23 | 0.43 | 0.48 | 0.05 | 0.43 | 0.40 | 0.14 |

Line timings barely moved, but pinyin makes the confidence usable as a quality
signal. Low-confidence chunks (< ~0.15) were the interlude-heavy 75 s chunk and
the outro, the spots to check by ear. On 異種 (romaji) confidence was 0.16–0.34.

## Open items

- Listen-check the `.lrc` files; build a small hand-labelled set for real
  error numbers.
- Per-kana / per-syllable timestamps (the CTC letter frames already have them).
- Cantonese vs Mandarin: pinyin worked for a Cantonese song; untested on
  Mandarin songs.

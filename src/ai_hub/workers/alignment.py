"""Qwen3-ForcedAligner worker. Run as `python -m ai_hub.workers.alignment`.

The API imports this module only for the job/result models; torch and the
model are imported inside `load()`, i.e. only in the worker process.
"""

import os
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from .protocol import serve

MODEL_ID = os.environ.get("AI_HUB_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
DEVICE = os.environ.get("AI_HUB_ALIGNER_DEVICE", "cuda:0")

# From the model config's `support_languages`. Only one language per sample:
# it picks the word tokenizer (Japanese and Korean have their own, everything
# else splits on spaces and treats each CJK character as a word, so mixed
# Chinese/English text works as "Chinese" or "English").
Language = Literal[
    "Chinese",
    "Cantonese",
    "English",
    "German",
    "Spanish",
    "French",
    "Italian",
    "Portuguese",
    "Russian",
    "Korean",
    "Japanese",
]

MAX_AUDIO_SECONDS = 180  # qwen_asr's MAX_FORCE_ALIGN_INPUT_SECONDS


class AlignJob(BaseModel):
    audio_path: str
    text: str
    language: Language


class AlignedWord(BaseModel):
    text: str
    start: float
    end: float


class AlignResult(BaseModel):
    duration: float
    words: list[AlignedWord]


def load() -> Callable[[AlignJob], AlignResult]:
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ForcedAligner

    aligner = Qwen3ForcedAligner.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE
    )

    def align(job: AlignJob) -> AlignResult:
        audio, sr = sf.read(job.audio_path, dtype="float32", always_2d=False)
        duration = len(audio) / sr
        if duration > MAX_AUDIO_SECONDS:
            msg = f"audio is {duration:.1f}s, max is {MAX_AUDIO_SECONDS}s"
            raise ValueError(msg)
        [result] = aligner.align(
            audio=(audio, sr), text=job.text, language=job.language
        )
        return AlignResult(
            duration=round(duration, 3),
            words=[
                AlignedWord(text=it.text, start=it.start_time, end=it.end_time)
                for it in result
            ],
        )

    return align


if __name__ == "__main__":
    serve(AlignJob, load)

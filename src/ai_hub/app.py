import logging
import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .workers.alignment import AlignedWord, AlignJob, AlignResult, Language
from .workers.manager import WorkerCrashed, WorkerError, WorkerManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

IDLE_TIMEOUT = float(os.environ.get("AI_HUB_IDLE_TIMEOUT", "300"))
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024

aligner = WorkerManager[AlignJob, AlignResult](
    "alignment", "ai_hub.workers.alignment", AlignResult, idle_timeout=IDLE_TIMEOUT
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    yield
    await aligner.shutdown()


app = FastAPI(title="ai-hub", lifespan=lifespan)


class AlignmentRequest(BaseModel):
    audio_url: str
    text: str = Field(min_length=1)
    language: Language

    @field_validator("language", mode="before")
    @classmethod
    def _capitalize(cls, v: object) -> object:
        return v.strip().capitalize() if isinstance(v, str) else v


class AlignmentResponse(BaseModel):
    language: Language
    duration: float
    words: list[AlignedWord]


async def download(url: str, dest: str) -> None:
    try:
        async with (
            httpx.AsyncClient(follow_redirects=True, timeout=60) as client,
            client.stream("GET", url) as resp,
        ):
            _ = resp.raise_for_status()
            size = 0
            async with await anyio.open_file(dest, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise HTTPException(413, "audio file too large")
                    _ = await f.write(chunk)
    except httpx.HTTPError as e:
        raise HTTPException(400, f"could not download audio_url: {e}") from e


@app.post("/v1/audio/alignments", response_model=AlignmentResponse)
async def create_alignment(req: AlignmentRequest) -> AlignmentResponse:
    with tempfile.TemporaryDirectory(prefix="ai-hub-") as tmp:
        audio_path = os.path.join(tmp, "audio")
        await download(req.audio_url, audio_path)
        job = AlignJob(audio_path=audio_path, text=req.text, language=req.language)
        try:
            result = await aligner.run(job)
        except WorkerError as e:
            raise HTTPException(422, str(e)) from e
        except WorkerCrashed as e:
            raise HTTPException(503, str(e)) from e
    return AlignmentResponse(
        language=req.language, duration=result.duration, words=result.words
    )


@app.get("/health")
async def health() -> dict[str, object]:
    return {"workers": {aligner.name: aligner.status()}}

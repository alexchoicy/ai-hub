# ai-hub

One API for local AI services. Each model runs in its own subprocess. The subprocess
starts on the first request, handles one job at a time, and exits after
`AI_HUB_IDLE_TIMEOUT` seconds with no requests (default 300), so its VRAM is freed.

```sh
uv run ai-hub   # AI_HUB_HOST / AI_HUB_PORT, default 0.0.0.0:8000
```

## POST /v1/audio/alignments

Forced alignment with Qwen3-ForcedAligner-0.6B. Audio can be at most 180 s long.

```json
{"audio_url": "https://.../a.wav", "text": "transcript", "language": "English"}
```

`language` (case-insensitive) must be one of: Chinese, Cantonese, English, German,
Spanish, French, Italian, Portuguese, Russian, Korean, Japanese.

Response: `{"language", "duration", "words": [{"text", "start", "end"}]}` (times in seconds).

`GET /health` shows whether each worker is running or busy.

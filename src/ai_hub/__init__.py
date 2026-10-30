def main() -> None:
    import os

    import uvicorn

    uvicorn.run(
        "ai_hub.app:app",
        host=os.environ.get("AI_HUB_HOST", "0.0.0.0"),
        port=int(os.environ.get("AI_HUB_PORT", "8000")),
    )

"""Wire protocol between the API and a worker subprocess.

The API writes one job per line to the worker's stdin as JSON. The worker
answers with one `Reply` per line on stdout: first a `ready` reply once the
model is loaded, then one reply per job. Logs go to stderr.
"""

import os
import sys
import traceback
from collections.abc import Callable
from typing import TextIO

from pydantic import BaseModel, JsonValue


class Reply(BaseModel):
    ready: bool = False
    ok: bool = False
    result: JsonValue = None
    error: str | None = None


def serve[JobT: BaseModel](
    job_type: type[JobT],
    load: Callable[[], Callable[[JobT], BaseModel]],
) -> None:
    """Worker side: load the model, then answer jobs until stdin closes."""
    out = _take_stdout()
    try:
        handle = load()
    except Exception:  # noqa: BLE001 - report any load failure to the API
        _send(out, Reply(error=traceback.format_exc()))
        sys.exit(1)
    _send(out, Reply(ready=True))

    for line in sys.stdin:
        try:
            result = handle(job_type.model_validate_json(line))
            reply = Reply(ok=True, result=result.model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001 - a bad job must not kill the worker
            reply = Reply(error=f"{type(e).__name__}: {e}")
        _send(out, reply)


def _take_stdout() -> TextIO:
    """Keep the real stdout for replies; send everything else printed to stderr."""
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    _ = os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return out


def _send(out: TextIO, reply: Reply) -> None:
    _ = out.write(reply.model_dump_json() + "\n")

"""Run a model in a subprocess that is started on demand and stopped when idle.

The subprocess owns the GPU memory, so ending it is the only reliable way to
give all VRAM back. Jobs are serialized: one runs at a time per worker, others
wait on the lock.
"""

import asyncio
import logging
import sys
import time
from asyncio.subprocess import PIPE, Process

from pydantic import BaseModel

from .protocol import Reply

logger = logging.getLogger(__name__)

_LINE_LIMIT = 64 * 1024 * 1024  # max size of one reply


class WorkerError(RuntimeError):
    """The worker ran the job and reported a failure."""


class WorkerCrashed(RuntimeError):
    """The worker died, hung, or never became ready."""


class WorkerManager[JobT: BaseModel, ResultT: BaseModel]:
    def __init__(
        self,
        name: str,
        module: str,
        result_type: type[ResultT],
        idle_timeout: float,
        job_timeout: float = 300,
        startup_timeout: float = 600,
    ):
        self.name: str = name
        self._module: str = module
        self._result_type: type[ResultT] = result_type
        self._idle_timeout: float = idle_timeout
        self._job_timeout: float = job_timeout
        self._startup_timeout: float = startup_timeout
        self._lock: asyncio.Lock = asyncio.Lock()
        self._proc: Process | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._last_used: float = 0.0

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def status(self) -> dict[str, object]:
        idle_for = time.monotonic() - self._last_used
        return {
            "running": self.running,
            "busy": self._lock.locked(),
            "idle_timeout": self._idle_timeout,
            "idle_for": round(idle_for, 1) if self.running else None,
        }

    async def run(self, job: JobT) -> ResultT:
        async with self._lock:
            if self._idle_task is not None:
                _ = self._idle_task.cancel()
            try:
                return await self._run(job)
            except WorkerError:
                raise
            except BaseException:
                # Timeout, crash or cancellation: the worker may still be busy
                # or have a stale reply buffered, so never reuse it.
                self._kill()
                raise
            finally:
                self._last_used = time.monotonic()
                if self.running:
                    self._idle_task = asyncio.create_task(self._stop_when_idle())

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            _ = self._idle_task.cancel()
        async with self._lock:
            await self._stop()

    async def _run(self, job: JobT) -> ResultT:
        if not self.running:
            await self._start()
        proc = self._proc
        assert proc is not None and proc.stdin is not None

        proc.stdin.write(job.model_dump_json().encode() + b"\n")
        await proc.stdin.drain()
        reply = await self._read_reply(self._job_timeout)
        if not reply.ok:
            raise WorkerError(reply.error or "unknown error")
        return self._result_type.model_validate(reply.result)

    async def _start(self) -> None:
        logger.info("[%s] starting worker", self.name)
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            self._module,
            stdin=PIPE,
            stdout=PIPE,
            limit=_LINE_LIMIT,
        )
        reply = await self._read_reply(self._startup_timeout)
        if not reply.ready:
            raise WorkerCrashed(f"{self.name} failed to start:\n{reply.error}")
        logger.info("[%s] worker ready (pid %s)", self.name, self._proc.pid)

    async def _read_reply(self, timeout: float) -> Reply:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout)
        except TimeoutError:
            raise WorkerCrashed(f"{self.name} gave no answer in {timeout}s") from None
        if not line:
            code = await proc.wait()
            raise WorkerCrashed(f"{self.name} exited unexpectedly (exit code {code})")
        return Reply.model_validate_json(line)

    async def _stop(self) -> None:
        """Close stdin so the worker exits on its own; kill it if it doesn't."""
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        assert proc.stdin is not None
        proc.stdin.close()
        try:
            _ = await asyncio.wait_for(proc.wait(), 10)
        except TimeoutError:
            proc.kill()
            _ = await proc.wait()
        logger.info("[%s] worker stopped", self.name)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            proc.kill()
            logger.warning("[%s] worker killed", self.name)

    async def _stop_when_idle(self) -> None:
        await asyncio.sleep(self._idle_timeout)
        async with self._lock:
            if self.running:
                logger.info(
                    "[%s] idle for %ss, stopping", self.name, self._idle_timeout
                )
                await self._stop()

"""Liveness heartbeat for long-running background jobs.

A worker that dies abruptly (OOM, forced kill, hardware failure) leaves its
claimed file row in a busy status forever from the database's point of view.
Every long job runs inside `Heartbeat`, which stamps the row's
last_heartbeat_at every INTERVAL seconds from a daemon thread; the sweeper
(app.main) reclaims any busy row whose heartbeat has been silent longer than
STALE_AFTER. A live 2-hour extraction is never touched - it keeps
heartbeating - while a dead worker's row frees in about 3 minutes.
"""
import logging
import threading

from app.db import liveness_pool

logger = logging.getLogger(__name__)


def _stamp(file_id: str) -> None:
    # dedicated pool: a stamp can never wait behind extraction/ingest queries
    with liveness_pool.connection() as conn:
        conn.execute(
            "UPDATE files SET last_heartbeat_at = now() WHERE id = %s", (file_id,)
        )

# Stamp cadence vs reclaim threshold: ~4 missed beats before a row is
# declared dead, so one slow database write never kills a healthy job.
INTERVAL_SECONDS = 45
STALE_AFTER = "3 minutes"  # SQL interval used by the sweeper


class Heartbeat:
    """Context manager: stamps files.last_heartbeat_at while the body runs."""

    def __init__(self, files_repo, file_id: str):
        self._files = files_repo
        self._file_id = file_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(INTERVAL_SECONDS):
            try:
                _stamp(self._file_id)
            except Exception:  # a failed stamp must never kill the job
                logger.warning("heartbeat write failed for %s", self._file_id)

    def __enter__(self) -> "Heartbeat":
        try:
            _stamp(self._file_id)  # immediate first stamp
        except Exception:
            logger.warning("initial heartbeat failed for %s", self._file_id)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()

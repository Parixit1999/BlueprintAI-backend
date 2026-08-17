"""Bounded pools for background document jobs.

FastAPI's BackgroundTasks runs sync functions on the SAME thread pool that
serves sync request handlers. Queue 150 ingests there and every API request
waits behind minutes of embedding - the app reads as down (ALB 504s) while
working exactly as coded. These executors keep slow document work on its own
bounded threads: submissions queue in the executor, requests keep their pool.

The keepalive matters: a job waiting in an executor queue holds a DB row in a
busy status ('uploaded'/'ingesting') with no heartbeat, which the stale-work
sweeper would reclaim as dead. The sweeper's loop stamps every queued id each
minute, so a live queue never looks like a crashed worker - and if this
instance dies, the stamps stop and reclamation works exactly as designed.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

from app.config import settings

_lock = threading.Lock()
_queued: set[str] = set()

extract_pool = ThreadPoolExecutor(
    max_workers=settings.extract_concurrency, thread_name_prefix="extract"
)
ingest_pool = ThreadPoolExecutor(
    max_workers=settings.ingest_concurrency, thread_name_prefix="ingest"
)


def submit(pool: ThreadPoolExecutor, file_id: str, fn, *args) -> None:
    """Queue a job; the file id is keepalive-stamped until the job starts."""
    with _lock:
        _queued.add(file_id)

    def run():
        with _lock:
            _queued.discard(file_id)
        fn(*args)

    pool.submit(run)


def queued_ids() -> list[str]:
    with _lock:
        return list(_queued)

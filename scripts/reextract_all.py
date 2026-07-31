"""Re-run extraction over documents already in the archive.

Use this after an extraction-pipeline change (better component detection,
better boxes, better tagging) so existing documents get the new results
instead of only newly uploaded ones.

    python scripts/reextract_all.py                      # dry run - shows the plan, changes nothing
    python scripts/reextract_all.py --yes --limit 5      # a small batch first
    python scripts/reextract_all.py --yes                # everything
    python scripts/reextract_all.py --yes --status ingested
    python scripts/reextract_all.py --yes --file-id <uuid> --file-id <uuid>

WHAT THIS COSTS, read before running with --yes:

1. Re-extraction DELETES a document's knowledge-base chunks and returns it to
   'needs review' (repositories.release_ingest_claim). Confirmed regions only
   re-enter the knowledge base when a HUMAN confirms the review again - that
   checkpoint is the whole point of the product and this script cannot skip
   it. Between running this and re-reviewing, those documents answer nothing
   in chat.
   => Work in batches. Re-extract a batch, review it, then do the next one.
   Running this across an entire archive at once empties the knowledge base
   until every document has been reviewed again.

2. It costs real model spend, now more than before: component detection makes
   up to `component_detail_calls` extra vision calls per sheet plus one naming
   call. Estimate before you start - the dry run prints the sheet count it is
   about to re-read.

3. It is slow. Extraction is bounded by FileService._extract_slots (2 at a
   time per process) on purpose, so a large archive takes hours. Safe to
   interrupt: each document is independent, and anything already re-extracted
   stays re-extracted.

Runs the SAME code path as the Retry/Re-extract buttons in the UI, in-process
against whatever database and AI provider the environment points at - so point
it at production deliberately, not by accident.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")

from app.config import settings  # noqa: E402
from app.db import pool  # noqa: E402
from app.dependencies import file_service  # noqa: E402

# Documents worth re-reading. 'uploaded' is in-flight and 'ingesting' is being
# worked on by someone right now - taking those would fight the live worker.
RE_EXTRACTABLE = ("extracted", "ingested", "failed")


def plan(status_filter: str | None, file_ids: list[str], limit: int | None) -> list[dict]:
    service = file_service()
    documents = service.list_files()
    if file_ids:
        wanted = set(file_ids)
        documents = [d for d in documents if d["file_id"] in wanted]
    else:
        allowed = (status_filter,) if status_filter else RE_EXTRACTABLE
        documents = [d for d in documents if d["status"] in allowed]
    # oldest first: the archive's earliest documents were extracted by the
    # oldest pipeline, so they gain the most
    documents.sort(key=lambda d: d["created_at"])
    return documents[:limit] if limit else documents


def reextract_one(document: dict) -> tuple[str, str]:
    """Returns (filename, outcome). Never raises: one bad document must not
    end a run that still has hundreds to go."""
    service = file_service()
    file_id = document["file_id"]
    try:
        service.prepare_reextract(file_id)
    except Exception as exc:
        return document["filename"], f"skipped ({type(exc).__name__}: {exc})"
    try:
        # run_matcher=None, matching the Re-extract route: the document's
        # drawing assignment already exists and must not be second-guessed
        service.process_upload(file_id, None)
    except Exception as exc:  # process_upload records failures on the row too
        return document["filename"], f"failed ({type(exc).__name__}: {exc})"
    return document["filename"], "re-extracted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true",
                        help="actually re-extract; without it this is a dry run")
    parser.add_argument("--status", choices=RE_EXTRACTABLE,
                        help="only documents in this status (default: all re-extractable)")
    parser.add_argument("--file-id", action="append", default=[],
                        help="specific document id; repeatable")
    parser.add_argument("--limit", type=int, help="stop after this many documents")
    parser.add_argument("--workers", type=int, default=2,
                        help="documents in flight (default 2, matching the "
                             "extraction semaphore - higher will just queue)")
    args = parser.parse_args()

    pool.open()
    documents = plan(args.status, args.file_id, args.limit)
    if not documents:
        print("Nothing to re-extract.")
        return 0

    ingested = sum(1 for d in documents if d["status"] == "ingested")
    print(f"{len(documents)} document(s) to re-extract "
          f"(provider={settings.ai_provider}, "
          f"detail calls/sheet={settings.component_detail_calls})")
    for d in documents[:20]:
        print(f"  {d['status']:<10} {d['chunk_count']:>5} chunks  {d['filename']}")
    if len(documents) > 20:
        print(f"  ... and {len(documents) - 20} more")

    if ingested:
        print(f"\n!! {ingested} of these are currently INGESTED. Re-extracting drops "
              f"their\n   knowledge-base chunks and sends them back to review - they "
              f"answer\n   nothing in chat until a human confirms each one again.")
    if not args.yes:
        print("\nDry run. Re-run with --yes to do it (--limit N for a batch first).")
        return 0

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as workers:
        for filename, outcome in workers.map(reextract_one, documents):
            done += 1
            print(f"[{done}/{len(documents)}] {outcome}: {filename}", flush=True)

    print(f"\nDone. Review the re-extracted documents and confirm each one to "
          f"put it back in the knowledge base.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

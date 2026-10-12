"""Enqueue ingestion steps on Hopper as http jobs that call sift-ingest.

Every step has an idempotency key derived from what it works on, so enqueuing the same step
twice (a retry, two last-batch finishers) creates one job.
"""

from typing import Any

import httpx

from sift.config import get_settings

# Seconds Hopper waits for sift-ingest's answer. Parsing a large page or PDF, or embedding
# while sift-models is busy with other batches, can take far longer than Hopper's default 10.
TIMEOUTS = {"parse": 120, "embed": 120, "activate": 30}
# Embedding gets its own queue, drained by Hopper workers sized to sift-models (one model call
# at a time). On the shared queue, 60 worker slots fired 60 embed calls at once into that one
# model server; calls queued past their timeout and the retries made it worse. The backlog
# belongs in Hopper's queue, not in open HTTP connections.
QUEUES = {"parse": "default", "embed": "sift-embed", "activate": "default"}


class HopperBusy(Exception):
    """Hopper refused or was unreachable (429, 503, network). Retry later."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


async def enqueue_step(
    client: httpx.AsyncClient, step: str, body: dict[str, Any], key: str
) -> None:
    s = get_settings()
    request = {
        "task": "http",
        "queue": QUEUES[step],
        "payload": {
            "url": f"{s.ingest_url}/internal/{step}",
            "method": "POST",
            "body": body,
            "timeout": TIMEOUTS[step],
        },
    }
    try:
        response = await client.post(
            f"{s.hopper_url}/v1/jobs",
            json=request,
            headers={
                "Authorization": f"Bearer {s.hopper_api_key.get_secret_value()}",
                "Idempotency-Key": key,
            },
        )
    except httpx.TransportError as exc:
        raise HopperBusy(f"Hopper unreachable: {type(exc).__name__}") from exc
    if response.status_code in (200, 201):  # 200: the same key again, the job already exists
        return
    if response.status_code in (429, 503):
        after = response.headers.get("retry-after")
        raise HopperBusy(f"Hopper answered {response.status_code}", float(after) if after else None)
    raise RuntimeError(f"Hopper refused the job ({response.status_code}): {response.text[:300]}")

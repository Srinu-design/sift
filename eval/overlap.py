"""Check HNSW recall overlap against exact brute-force scans on filtered questions.

python -m eval.overlap --split dev --k 50
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.metrics import mean
from eval.run import HERE, load_golden
from sift.config import get_settings
from sift.db import create_engine
from sift.ingest import pipeline
from sift.retrieval.dense import dense
from sift.retrieval.filters import Filters


async def check_overlap(args: argparse.Namespace) -> dict[str, Any]:
    golden_path: Path = args.golden
    questions = load_golden(golden_path, args.split, args.tenant)
    filtered = [q for q in questions if q.get("filters")]

    vectors: list[list[float]] = []
    if filtered:
        from sift.models_svc.encoder import encoder

        vectors = list(encoder().embed([q["question"] for q in filtered], "query"))

    engine = create_engine(get_settings())
    tenant_id, collection_id = await pipeline.ensure_collection(engine, args.tenant, args.tenant)

    per_question: list[dict[str, Any]] = []
    overlaps: list[float] = []
    short_lists = 0
    for q, vec in zip(filtered, vectors, strict=True):
        filters = Filters.model_validate(q["filters"])
        hnsw_hits = await dense(
            engine,
            tenant_id=tenant_id,
            collection_id=collection_id,
            query_vector=vec,
            filters=filters,
            k=args.k,
            scan="hnsw",
        )
        exact_hits = await dense(
            engine,
            tenant_id=tenant_id,
            collection_id=collection_id,
            query_vector=vec,
            filters=filters,
            k=args.k,
            scan="exact",
        )
        hnsw_ids = {h.chunk_id for h in hnsw_hits}
        exact_ids = {h.chunk_id for h in exact_hits}
        overlap = len(hnsw_ids & exact_ids) / len(exact_ids) if exact_ids else 1.0
        overlaps.append(overlap)
        if len(hnsw_hits) < len(exact_hits):
            short_lists += 1
        per_question.append(
            {
                "id": q["id"],
                "filters": q["filters"],
                "hnsw_count": len(hnsw_hits),
                "exact_count": len(exact_hits),
                "overlap": round(overlap, 4),
            }
        )
    await engine.dispose()

    return {
        "split": args.split,
        "golden": golden_path.stem,
        "tenant": args.tenant,
        "k": args.k,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n": len(per_question),
        "mean_overlap": round(mean(overlaps), 4) if overlaps else 0.0,
        "min_overlap": round(min(overlaps), 4) if overlaps else 0.0,
        "short_lists": short_lists,
        "per_question": per_question,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.overlap",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--golden", type=Path, default=HERE / "golden/v1.jsonl")
    p.add_argument("--tenant", default="pgdocs")
    args = p.parse_args()
    report = asyncio.run(check_overlap(args))
    print(
        f"overlap | split {report['split']} | k={report['k']} | n={report['n']} | "
        f"mean={report['mean_overlap']:.3f} | min={report['min_overlap']:.3f} | "
        f"short={report['short_lists']}"
    )

    results = HERE / "results"
    results.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = results / f"{stamp}-overlap-{report['split']}-{report['golden']}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"overlap results: {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

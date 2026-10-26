"""Run the golden set through a retrieval strategy and report hit@k, recall@k and MRR@10.

    python -m eval.run --strategy naive --split dev
    python -m eval.run --strategy naive --split dev --write-baseline

Embeds the questions in-process with the pinned local model (no paid API, deterministic),
retrieves against the ingested corpus, and writes every question's ranking and scores to
eval/results/, named with the date, strategy, split and golden-set version. Retrieval
metrics count answerable questions only; refusal metrics are eval.answers' job.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from eval import HERE, KS, git_commit, metric_names, percentile, save
from eval.metrics import (
    Gold,
    Retrieved,
    bootstrap_ci,
    first_hit_rank,
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from sift.config import get_settings
from sift.db import create_engine
from sift.ingest import chunker, pipeline
from sift.retrieval.filters import Filters
from sift.retrieval.rerank import local_reranker
from sift.retrieval.search import DEPTH, STRATEGIES, search


def load_golden(path: Path, split: str, tenant: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["tenant"] == tenant and (split == "all" or r["split"] == split)]
    for r in rows:
        try:
            Filters.model_validate(r.get("filters") or {})
        except ValidationError as exc:
            raise SystemExit(f"{r['id']}: {exc}") from exc
    return rows


def docs_by_url() -> dict[str, str]:
    """Golden doc id for every corpus url, from every tenant's manifest."""
    return {
        m["url"]: m["doc"]
        for path in sorted((HERE.parent / "corpus").glob("*manifest.jsonl"))
        for m in map(json.loads, path.read_text().splitlines())
    }


async def find_collection(engine: AsyncEngine, tenant: str, collection: str) -> tuple[UUID, UUID]:
    """(tenant id, collection id). Never creates them: a typo would score an empty index as 0."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT c.tenant_id, c.id FROM collections c JOIN tenants t"
                    " ON t.id = c.tenant_id WHERE t.name = :t AND c.name = :c"
                ),
                {"t": tenant, "c": collection},
            )
        ).first()
    if row is None:
        await engine.dispose()  # exiting: close the pool so asyncpg does not warn
        raise SystemExit(f"no collection {collection!r} in tenant {tenant!r}: ingest it first")
    return row[0], row[1]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    golden_path: Path = args.golden
    questions = load_golden(golden_path, args.split, args.tenant)
    answerable = [q for q in questions if q["answerable"]]
    doc_of_url = docs_by_url()
    vectors: list[list[float] | None]
    if args.strategy == "lexical":
        vectors = [None] * len(answerable)
    else:
        from sift.models_svc.encoder import encoder

        vectors = list(encoder().embed([q["question"] for q in answerable], "query"))

    engine = create_engine(get_settings())
    tenant_id, collection_id = await find_collection(
        engine, args.tenant, args.collection or args.tenant
    )
    async with engine.connect() as conn:
        index_version = await conn.scalar(
            text("SELECT index_version FROM collections WHERE id = :c"), {"c": collection_id}
        )

    reranker = local_reranker() if args.strategy == "hybrid_rerank" else None
    per_question: list[dict[str, Any]] = []
    for q, vec in zip(answerable, vectors, strict=True):
        filters = Filters() if args.no_filters else Filters.model_validate(q.get("filters") or {})
        res = await search(
            engine,
            tenant_id=tenant_id,
            collection_id=collection_id,
            strategy=args.strategy,
            question=q["question"],
            query_vector=vec,
            filters=filters,
            k=DEPTH,
            depth=DEPTH,
            lexical_mode=args.lexical_mode,
            reranker=reranker,
            rerank_depth=args.rerank_depth,
        )
        hits = res.hits
        ranked = [
            Retrieved(
                doc_of_url.get(h.source_uri, h.source_uri), h.anchor_path, h.page_start, h.page_end
            )
            for h in hits
        ]
        golds = [Gold(g["doc"], g.get("section"), g.get("page")) for g in q["gold"]]
        scores = {f"hit@{k}": hit_at_k(ranked, golds, k) for k in KS}
        scores |= {f"recall@{k}": recall_at_k(ranked, golds, k) for k in KS}
        scores["mrr@10"] = reciprocal_rank(ranked, golds, 10)
        per_question.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "gold": q["gold"],
                "first_hit_rank": first_hit_rank(ranked, golds),
                "scores": scores,
                "timings_ms": res.timings_ms,
                "top": [
                    {
                        "doc": r.doc,
                        "anchor": h.anchor,
                        "heading": h.heading,
                        "score": round(h.score, 4),
                    }
                    for r, h in zip(ranked[:10], hits[:10], strict=True)
                ],
            }
        )
    await engine.dispose()

    def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(rows)}
        for name in metric_names():
            values = [r["scores"][name] for r in rows]
            low, high = bootstrap_ci(values)
            out[name] = {"mean": round(mean(values), 4), "ci95": [round(low, 4), round(high, 4)]}
        return out

    stages = list(per_question[0]["timings_ms"]) if per_question else []
    timings_ms = {
        stage: {
            "p50": percentile([r["timings_ms"][stage] for r in per_question], 0.50),
            "p95": percentile([r["timings_ms"][stage] for r in per_question], 0.95),
        }
        for stage in stages
    }
    strategy_label = (
        f"{args.strategy}-and"
        if args.lexical_mode == "and" and args.strategy != "naive"
        else args.strategy
    )
    if args.no_filters:
        strategy_label += "-nofilter"
    if args.strategy == "hybrid_rerank" and args.rerank_depth != get_settings().rerank_depth:
        strategy_label += f"-r{args.rerank_depth}"
    types = sorted({r["type"] for r in per_question})
    settings = get_settings()
    return {
        "strategy": strategy_label,
        "split": args.split,
        "golden": golden_path.stem,
        "tenant": args.tenant,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "setup": {
            "embed_model": pipeline.embed_model_key(),
            "rerank_model": f"{settings.rerank_model}@{settings.rerank_revision[:12]}",
            "chunker": chunker.NAME,
            "index_version": index_version,
            "depth": DEPTH,
            "rerank_depth": args.rerank_depth,
            "lexical_mode": args.lexical_mode,
        },
        "questions": len(questions),
        "unanswerable_skipped": len(questions) - len(answerable),
        "timings_ms": timings_ms,
        "overall": summarise(per_question),
        "by_type": {t: summarise([r for r in per_question if r["type"] == t]) for t in types},
        "per_question": per_question,
    }


def table(report: dict[str, Any]) -> str:
    cols = ["hit@5", "hit@8", "recall@10", "mrr@10"]
    lines = [
        f"{report['strategy']} | golden {report['golden']} | split {report['split']} | "
        f"{report['overall']['n']} answerable (+{report['unanswerable_skipped']} unanswerable "
        f"skipped) | {report['setup']['chunker']} | {report['setup']['embed_model']}",
        f"{'':<16}{'n':>4}" + "".join(f"{c:>22}" for c in cols),
    ]
    rows = [("overall", report["overall"]), *report["by_type"].items()]
    for name, s in rows:
        cells = "".join(
            f"{s[c]['mean']:>8.3f} [{s[c]['ci95'][0]:.2f}, {s[c]['ci95'][1]:.2f}]" for c in cols
        )
        lines.append(f"{name:<16}{s['n']:>4}{cells}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--strategy", choices=STRATEGIES, default="naive")
    p.add_argument("--lexical-mode", choices=["or", "and"], default="or")
    p.add_argument(
        "--rerank-depth",
        type=int,
        default=get_settings().rerank_depth,
        help="fused hits sent to the reranker",
    )
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--golden", type=Path, default=HERE / "golden/v1.jsonl")
    p.add_argument("--tenant", default="pgdocs")
    p.add_argument("--collection", help="defaults to the tenant name")
    p.add_argument("--no-filters", action="store_true", help="ablation: ignore question filters")
    p.add_argument("--write-baseline", action="store_true")
    args = p.parse_args()
    report = asyncio.run(run(args))
    print(table(report))

    out = save(report, f"{report['strategy']}-{report['split']}-{report['golden']}")
    print(f"per-question results: {out.relative_to(HERE.parent)}")
    if args.write_baseline:
        baseline_path = HERE / "baselines/retrieval.json"
        baseline_path.parent.mkdir(exist_ok=True)
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        baseline[report["strategy"]] = {k: v for k, v in report.items() if k != "per_question"}
        baseline_path.write_text(json.dumps(baseline, indent=1, sort_keys=True) + "\n")
        print(f"baseline written: {baseline_path.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

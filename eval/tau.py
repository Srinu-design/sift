"""Tune the gate-1 retrieval refusal threshold (tau) on a split.

python -m eval.tau --split dev --max-wrong 0.05
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval import HERE, git_commit, save
from eval.run import find_collection, load_golden
from sift.config import get_settings
from sift.db import create_engine
from sift.models_svc.encoder import encoder
from sift.retrieval.filters import Filters
from sift.retrieval.rerank import local_reranker
from sift.retrieval.search import DEPTH, search


def evaluate_tau(rows: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    ans_refused = sum(1 for r in ans if r["best_score"] < tau)
    unans_refused = sum(1 for r in unans if r["best_score"] < tau)

    wrong_refusal_rate = ans_refused / len(ans) if ans else 0.0
    refusal_recall = unans_refused / len(unans) if unans else 0.0
    return {
        "tau": round(tau, 6),
        "refusal_recall": round(refusal_recall, 4),
        "wrong_refusal_rate": round(wrong_refusal_rate, 4),
        "confusion_matrix": {
            "answerable_answered": len(ans) - ans_refused,
            "answerable_refused": ans_refused,
            "unanswerable_answered": len(unans) - unans_refused,
            "unanswerable_refused": unans_refused,
        },
    }


async def tune_tau(args: argparse.Namespace) -> dict[str, Any]:
    golden_path: Path = args.golden
    questions = load_golden(golden_path, args.split, args.tenant)
    settings = get_settings()

    vectors = list(encoder().embed([q["question"] for q in questions], "query"))
    reranker = local_reranker()
    engine = create_engine(settings)
    tenant_id, collection_id = await find_collection(engine, args.tenant, args.tenant)

    per_question: list[dict[str, Any]] = []
    for q, vec in zip(questions, vectors, strict=True):
        filters = Filters.model_validate(q.get("filters") or {})
        res = await search(
            engine,
            tenant_id=tenant_id,
            collection_id=collection_id,
            strategy="hybrid_rerank",
            question=q["question"],
            query_vector=vec,
            filters=filters,
            k=settings.context_k,
            depth=DEPTH,
            rerank_depth=settings.rerank_depth,
            reranker=reranker,
        )
        best = res.hits[0].score if res.hits else 0.0
        per_question.append(
            {
                "id": q["id"],
                "type": q["type"],
                "answerable": bool(q["answerable"]),
                "best_score": round(best, 6),
            }
        )
    await engine.dispose()

    candidates = sorted({0.0, *(float(r["best_score"]) for r in per_question)})
    evaluated = [evaluate_tau(per_question, t) for t in candidates]
    eligible = [c for c in evaluated if c["wrong_refusal_rate"] <= args.max_wrong]
    chosen = eligible[-1] if eligible else evaluated[0]

    return {
        "split": args.split,
        "golden": golden_path.stem,
        "tenant": args.tenant,
        "max_wrong": args.max_wrong,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "chosen_tau": chosen["tau"],
        "refusal_recall": chosen["refusal_recall"],
        "wrong_refusal_rate": chosen["wrong_refusal_rate"],
        "confusion_matrix": chosen["confusion_matrix"],
        "candidates": evaluated,
        "per_question": per_question,
    }


def format_report(report: dict[str, Any]) -> str:
    evaluated: list[dict[str, Any]] = report["candidates"]
    chosen_tau = report["chosen_tau"]
    if len(evaluated) <= 10:
        shown = evaluated
    else:
        step = max(1, (len(evaluated) - 1) // 9)
        indices = set(range(0, len(evaluated), step)) | {len(evaluated) - 1}
        shown = [c for idx, c in enumerate(evaluated) if idx in indices or c["tau"] == chosen_tau]

    lines = [
        f"tau tuning | golden {report['golden']} | split {report['split']} | "
        f"max_wrong <= {report['max_wrong']:.3f}",
        f"{'tau':>10} | {'refusal_recall':>14} | {'wrong_refusal':>13} | {'note':<6}",
    ]
    for c in shown:
        note = "<-- chosen" if c["tau"] == chosen_tau else ""
        lines.append(
            f"{c['tau']:>10.4f} | {c['refusal_recall']:>14.3f} | "
            f"{c['wrong_refusal_rate']:>13.3f} | {note}"
        )

    cm = report["confusion_matrix"]
    lines.extend(
        [
            f"\nchosen tau: {chosen_tau:.4f} "
            f"(refusal_recall={report['refusal_recall']:.3f}, "
            f"wrong_refusal_rate={report['wrong_refusal_rate']:.3f})",
            "confusion matrix (rows: actual, cols: answered / refused):",
            f"  answerable   : answered={cm['answerable_answered']:>4}  "
            f"refused={cm['answerable_refused']:>4}",
            f"  unanswerable : answered={cm['unanswerable_answered']:>4}  "
            f"refused={cm['unanswerable_refused']:>4}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.tau",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--max-wrong", type=float, default=0.05)
    p.add_argument("--golden", type=Path, default=HERE / "golden/v1.jsonl")
    p.add_argument("--tenant", default="pgdocs")
    args = p.parse_args()

    report = asyncio.run(tune_tau(args))
    print(format_report(report))

    out = save(report, f"tau-{report['split']}-{report['golden']}")
    print(f"tau results: {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

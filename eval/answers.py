"""Run the end-to-end answer evaluation with resumable JSONL output and rate limiting.

python -m eval.answers --split dev --strategy hybrid_rerank --rpm 8
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from eval import HERE, git_commit, percentile, read_jsonl, save
from eval.metrics import Gold, Retrieved, is_hit
from eval.run import docs_by_url, find_collection, load_golden
from sift.answer.generate import AnswerResult, answer
from sift.answer.llm import LLMUnavailable, make_llm
from sift.answer.prompt import PROMPT_VERSION, source_label
from sift.answer.refusal import gate1
from sift.config import Settings, get_settings
from sift.db import create_engine
from sift.retrieval.filters import Filters
from sift.retrieval.rerank import local_reranker
from sift.retrieval.search import DEPTH, STRATEGIES, search


def summarise_answers(
    rows: list[dict[str, Any]],
    *,
    settings: Settings,
    split: str,
    golden: str,
    strategy: str,
    tau: float | None,
) -> dict[str, Any]:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    ans_answered = sum(1 for r in ans if not r["refused"])
    ans_refused = sum(1 for r in ans if r["refused"])
    unans_answered = sum(1 for r in unans if not r["refused"])
    unans_refused = sum(1 for r in unans if r["refused"])

    total_refused = ans_refused + unans_refused
    refusal_precision = unans_refused / total_refused if total_refused else 0.0
    refusal_recall = unans_refused / len(unans) if unans else 0.0
    wrong_refusal_rate = ans_refused / len(ans) if ans else 0.0

    refusals_by_gate = {
        gate: sum(1 for r in rows if r["refused"] and r.get("refusal_gate") == gate)
        for gate in ("retrieval", "model", "citation")
    }

    ans_cit_hits = [hit for r in ans if not r["refused"] for hit in r.get("citation_hits", [])]
    citation_accuracy = (
        sum(1 for h in ans_cit_hits if h) / len(ans_cit_hits) if ans_cit_hits else 0.0
    )

    answered_ans = [r for r in ans if not r["refused"]]
    with_gold = sum(1 for r in answered_ans if any(r.get("citation_hits", [])))
    answered_with_gold_share = with_gold / len(answered_ans) if answered_ans else 0.0

    attempts_hist = {str(k): sum(1 for r in rows if r.get("attempts") == k) for k in (0, 1, 2)}
    tokens_in = sum(int(r.get("tokens_in", 0)) for r in rows)
    tokens_out = sum(int(r.get("tokens_out", 0)) for r in rows)
    cost_usd = round(
        (tokens_in * settings.llm_price_in_per_mtok + tokens_out * settings.llm_price_out_per_mtok)
        / 1_000_000.0,
        6,
    )
    llm_times = [float(r.get("llm_ms", 0.0)) for r in rows]
    model_name = settings.answer_model if settings.llm_provider == "gemini" else "stub"

    return {
        "strategy": strategy,
        "split": split,
        "golden": golden,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "tau": tau,
        "n": len(rows),
        "confusion_matrix": {
            "answerable_answered": ans_answered,
            "answerable_refused": ans_refused,
            "unanswerable_answered": unans_answered,
            "unanswerable_refused": unans_refused,
        },
        "refusals_by_gate": refusals_by_gate,
        "refusal_precision": round(refusal_precision, 4),
        "refusal_recall": round(refusal_recall, 4),
        "wrong_refusal_rate": round(wrong_refusal_rate, 4),
        "citation_accuracy": round(citation_accuracy, 4),
        "answered_with_gold_citation": round(answered_with_gold_share, 4),
        "citation_gate_refusals": refusals_by_gate["citation"],
        "attempts_histogram": attempts_hist,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "llm_ms": {
            "p50": percentile(llm_times, 0.50),
            "p95": percentile(llm_times, 0.95),
        },
        "injection": {
            "n": sum(1 for r in rows if "injection_ok" in r),
            "passed": sum(1 for r in rows if r.get("injection_ok")),
        },
    }


def format_summary(report: dict[str, Any]) -> str:
    cm = report["confusion_matrix"]
    gates = report["refusals_by_gate"]
    return "\n".join(
        [
            f"answers | {report['strategy']} | golden {report['golden']} | "
            f"split {report['split']} | n={report['n']} | model={report['model']}",
            f"confusion: ans_answered={cm['answerable_answered']} "
            f"ans_refused={cm['answerable_refused']} "
            f"unans_answered={cm['unanswerable_answered']} "
            f"unans_refused={cm['unanswerable_refused']}",
            f"refusal: precision={report['refusal_precision']:.3f} "
            f"recall={report['refusal_recall']:.3f} "
            f"wrong_refusal={report['wrong_refusal_rate']:.3f} "
            f"gates={gates}",
            f"citations: accuracy={report['citation_accuracy']:.3f} "
            f"answered_with_gold={report['answered_with_gold_citation']:.3f} "
            f"citation_gate_refusals={report['citation_gate_refusals']}",
            f"llm: attempts={report['attempts_histogram']} "
            f"tokens_in={report['tokens_in']} tokens_out={report['tokens_out']} "
            f"cost=${report['cost_usd']:.6f} "
            f"p50={report['llm_ms']['p50']:.1f}ms p95={report['llm_ms']['p95']:.1f}ms",
            f"injection: {report['injection']['passed']}/{report['injection']['n']} passed",
        ]
    )


async def run_eval(args: argparse.Namespace, out_path: Path) -> list[dict[str, Any]]:
    settings = get_settings()
    questions = load_golden(args.golden, args.split, args.tenant)
    if args.limit is not None:
        questions = questions[: args.limit]

    existing = read_jsonl(out_path)
    done_ids = {r["id"] for r in existing}
    pending = [q for q in questions if q["id"] not in done_ids]

    if not pending:
        return existing

    doc_of_url = docs_by_url()

    vectors: list[list[float] | None]
    if args.strategy == "lexical":
        vectors = [None] * len(pending)
    else:
        from sift.models_svc.encoder import encoder

        vectors = list(encoder().embed([q["question"] for q in pending], "query"))

    reranker = local_reranker() if args.strategy == "hybrid_rerank" else None
    tau = settings.refusal_tau if args.strategy == "hybrid_rerank" else None
    min_interval = 60.0 / args.rpm if args.rpm > 0 else 0.0
    last_llm_at: float | None = None
    questions_called = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings)
    tenant_id, collection_id = await find_collection(
        engine, args.tenant, args.collection or args.tenant
    )

    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as http:
            llm = make_llm(settings, http)
            with out_path.open("a", encoding="utf-8") as fp:
                for q, vec in zip(pending, vectors, strict=True):
                    filters = Filters.model_validate(q.get("filters") or {})
                    search_res = await search(
                        engine,
                        tenant_id=tenant_id,
                        collection_id=collection_id,
                        strategy=args.strategy,
                        question=q["question"],
                        query_vector=vec,
                        filters=filters,
                        k=settings.context_k,
                        depth=DEPTH,
                        rerank_depth=settings.rerank_depth,
                        reranker=reranker,
                    )
                    hits = search_res.hits
                    best_score = hits[0].score if hits else None
                    will_call_llm = not (tau is not None and gate1(best_score, tau))
                    if will_call_llm and args.max_calls is not None:
                        if questions_called >= args.max_calls:  # spend cap: stop, resume later
                            print(f"--max-calls {args.max_calls} reached", file=sys.stderr)
                            break
                        questions_called += 1

                    if will_call_llm and last_llm_at is not None and min_interval > 0:
                        elapsed = time.monotonic() - last_llm_at
                        if elapsed < min_interval:
                            await asyncio.sleep(min_interval - elapsed)

                    ans_res: AnswerResult | None = None
                    for retry in range(4):
                        try:
                            if will_call_llm:
                                last_llm_at = time.monotonic()
                            ans_res = await answer(
                                q["question"],
                                hits,
                                llm,
                                tau=tau,
                                context_k=settings.context_k,
                            )
                            break
                        except LLMUnavailable as exc:
                            if retry == 3:
                                raise SystemExit(
                                    f"{q['id']}: LLMUnavailable after 3 retries ({exc}); "
                                    f"rerun command to resume from {out_path}"
                                ) from exc
                            await asyncio.sleep(60.0)

                    assert ans_res is not None
                    context_hits = hits[: settings.context_k]
                    golds = [
                        Gold(g["doc"], g.get("section"), g.get("page")) for g in q.get("gold", [])
                    ]
                    cit_hits: list[bool] = []
                    for c in ans_res.citations:
                        hit = context_hits[c.n - 1]
                        ret = Retrieved(
                            doc_of_url.get(hit.source_uri, hit.source_uri),
                            hit.anchor_path,
                            hit.page_start,
                            hit.page_end,
                        )
                        cit_hits.append(any(is_hit(ret, g) for g in golds))

                    row: dict[str, Any] = {
                        "id": q["id"],
                        "type": q["type"],
                        "answerable": bool(q["answerable"]),
                        "refused": ans_res.refused,
                        "refusal_gate": ans_res.refusal_gate,
                        "refusal_reason": ans_res.refusal_reason,
                        "answer": ans_res.answer,
                        "citations": [
                            {
                                "n": c.n,
                                "url": c.url,
                                "section": c.section,
                                "version": c.version,
                            }
                            for c in ans_res.citations
                        ],
                        "best_score": round(best_score, 6) if best_score is not None else None,
                        "attempts": ans_res.attempts,
                        "tokens_in": ans_res.tokens_in,
                        "tokens_out": ans_res.tokens_out,
                        "llm_ms": ans_res.llm_ms,
                        "citation_hits": cit_hits,
                        # What the model saw, so the judge and hand labels need no database.
                        "sources": [
                            {"n": n, "label": source_label(h), "text": h.content}
                            for n, h in enumerate(context_hits, start=1)
                        ],
                    }
                    if "injection" in q:  # instructions planted in a source must change nothing
                        rule = q["injection"]
                        text_out = ans_res.answer.lower()
                        row["injection_ok"] = (
                            not (rule.get("must_answer") and ans_res.refused)
                            and not any(s.lower() in text_out for s in rule["must_not_contain"])
                            and any(cit_hits)
                        )
                    fp.write(json.dumps(row) + "\n")
                    fp.flush()
                    existing.append(row)
    finally:
        await engine.dispose()

    return existing


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.answers",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--strategy", choices=STRATEGIES, default="hybrid_rerank")
    p.add_argument("--rpm", type=float, default=8.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--report", action="store_true")
    p.add_argument("--golden", type=Path, default=HERE / "golden/v1.jsonl")
    p.add_argument("--tenant", default="pgdocs")
    p.add_argument("--collection", help="defaults to the tenant name")
    p.add_argument(
        "--max-calls", type=int, help="spend cap: questions that may call the LLM in this run"
    )
    args = p.parse_args()

    golden_stem = args.golden.stem
    out_path: Path = (
        args.out
        if args.out is not None
        else HERE
        / f"results/answers-{args.split}-{golden_stem}-{args.strategy}-{PROMPT_VERSION}.jsonl"
    )

    rows = read_jsonl(out_path) if args.report else asyncio.run(run_eval(args, out_path))
    settings = get_settings()
    tau = settings.refusal_tau if args.strategy == "hybrid_rerank" else None
    report = summarise_answers(
        rows,
        settings=settings,
        split=args.split,
        golden=golden_stem,
        strategy=args.strategy,
        tau=tau,
    )
    print(format_summary(report))

    summary_path = save(report, f"answers-{args.split}-{golden_stem}-{args.strategy}")
    print(f"summary written: {summary_path.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

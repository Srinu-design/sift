"""Hand-label answers, to validate the LLM judge (Cohen's kappa in eval/kappa.py).

    python -m eval.label eval/results/answers-dev-v1-hybrid_rerank-answer-v2.jsonl \\
        --golden eval/golden/v1.jsonl --by <your name>

Shows one answered question at a time: the question, the reference answer, every source the
model saw and its answer. Each label is appended to --out at once, so `q` (or Ctrl-C) loses
nothing and the next run carries on. The sample is seeded, so it is the same 40 every time.
"""

import argparse
import json
import random
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FAITHFUL = {"y": 1.0, "p": 0.5, "n": 0.0}
CORRECT: dict[str, float | None] = {"1": 1.0, "h": 0.5, "0": 0.0, "s": None}


class Quit(Exception):
    pass


def select_sample(
    rows: list[dict[str, Any]], *, n: int, seed: int, labeled_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    """The seeded sample of answered rows (same ids whatever the file order), minus labelled."""
    answered = sorted((r for r in rows if not r["refused"]), key=lambda r: r["id"])
    sample = random.Random(seed).sample(answered, min(n, len(answered)))
    return [r for r in sample if r["id"] not in (labeled_ids or set())]


def ask(prompt: str, choices: dict[str, Any]) -> Any:
    while True:
        reply = input(prompt).strip().lower()
        if reply == "q":
            raise Quit
        if reply in choices:
            return choices[reply]
        print(f"  answer one of: {', '.join(choices)} (or q to quit)")


def show(row: dict[str, Any], golden: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"{row['id']} ({row['type']})  QUESTION: {golden['question']}")
    print(f"REFERENCE: {golden.get('reference_answer') or '(unanswerable: no reference)'}")
    for s in row["sources"]:
        print(f"\n[S{s['n']}] {s['label']}")
        print(textwrap.fill(" ".join(s["text"].split()), width=100))
    print("\nANSWER:\n" + textwrap.fill(row["answer"], width=100) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m eval.label", description=__doc__)
    p.add_argument("answers", type=Path)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("eval/labels/human-v1.jsonl"))
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--by", default="srinu")
    args = p.parse_args()

    rows = [json.loads(line) for line in args.answers.read_text().splitlines() if line.strip()]
    golden = {
        g["id"]: g for g in map(json.loads, args.golden.read_text().splitlines()) if g.get("id")
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = (
        {json.loads(line)["id"] for line in args.out.read_text().splitlines() if line.strip()}
        if args.out.exists()
        else set()
    )
    todo = select_sample(rows, n=args.n, seed=args.seed, labeled_ids=done)
    total = len(todo) + len(done)
    try:
        for i, row in enumerate(todo, start=len(done) + 1):
            show(row, golden[row["id"]])
            faithful = ask(
                "Faithful? every claim supported by the sources: [y]es / [p]artly / [n]o: ",
                FAITHFUL,
            )
            correct = ask(
                "Correct? Grade against what the QUESTION asks (reference detail it did not ask"
                " for is optional): [1] / [h]alf / [0] / [s]kip: ",
                CORRECT,
            )
            note = input("Note (optional): ").strip()
            label = {
                "id": row["id"],
                "faithful": faithful,
                "correctness": correct,
                "note": note,
                "labeled_by": args.by,
                "labeled_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            with args.out.open("a") as fp:
                fp.write(json.dumps(label) + "\n")
            print(f"saved ({i}/{total})")
    except (Quit, KeyboardInterrupt, EOFError):
        print("\nstopped; run the same command to carry on", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

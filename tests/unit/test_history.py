import json
from pathlib import Path

from eval.history import history


def write(path: Path, name: str, body: dict[str, object]) -> None:
    (path / name).write_text(json.dumps(body))


def test_keeps_retrieval_and_answer_runs_in_time_order(tmp_path: Path) -> None:
    metric = {"mean": 0.5, "ci95": [0.4, 0.6]}
    write(
        tmp_path,
        "b.json",
        {
            "run_at": "2026-10-20T10:00:00+00:00",
            "commit": "abc1234",
            "strategy": "hybrid",
            "split": "dev",
            "golden": "v1",
            "overall": {m: metric for m in ("hit@5", "hit@8", "recall@10", "mrr@10")},
        },
    )
    write(
        tmp_path,
        "a.json",
        {
            "run_at": "2026-10-22T10:00:00+00:00",
            "commit": "def5678",
            "strategy": "hybrid_rerank",
            "split": "dev",
            "golden": "v1",
            "refusal_recall": 0.87,
            "refusal_precision": 0.76,
            "citation_accuracy": 0.71,
        },
    )
    write(tmp_path, "kappa.json", {"run_at": "2026-10-23T10:00:00+00:00", "kappa": 0.49})

    runs = history(tmp_path)

    assert [r["kind"] for r in runs] == ["retrieval", "answers"]
    assert runs[0]["metrics"]["mrr@10"] == 0.5
    assert runs[1]["metrics"] == {
        "refusal_recall": 0.87,
        "refusal_precision": 0.76,
        "citation_accuracy": 0.71,
    }


def test_the_real_results_folder_parses() -> None:
    runs = history()
    assert runs and all(r["run_at"] and r["metrics"] for r in runs)

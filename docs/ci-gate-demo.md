# The CI retrieval gate blocking a bad PR

Build guide, week 5: "CI blocks a deliberately bad PR". Recorded 2026-10-23.

**The PR:** [Srinu-design/sift#8](https://github.com/Srinu-design/sift/pull/8), "rerank only the
fused top 10 (cuts rerank latency ~3x)", a one-line change to `rerank_depth` in `config.py`.
Closed without merging.

**Why it is a good test:** on the numbers a reviewer reads first it looks like an improvement.
It cuts the slowest stage (ADR-0003) and *raises* hit@5 and MRR@10. It still loses recall where
the answer step needs it: the top 8 sources the model sees.

**What the `eval-retrieval` job printed** ([run 39858931785](https://github.com/Srinu-design/sift/actions/runs/39858931785/job/120104273090)),
dev split, golden v1, gated strategy `hybrid_rerank`:

| strategy | metric | baseline | now | change | status |
| --- | --- | ---: | ---: | ---: | --- |
| hybrid_rerank | hit@5 | 0.887 | 0.915 | +0.028 | pass |
| hybrid_rerank | hit@8 | 0.986 | 0.958 | -0.028 | fail |
| hybrid_rerank | recall@10 | 0.979 | 0.951 | -0.028 | fail |
| hybrid_rerank | mrr@10 | 0.841 | 0.844 | +0.004 | pass |
| hybrid_rerank | paraphrase:hit@8 | 1.000 | 0.875 | -0.125 | fail |
| hybrid_rerank | paraphrase:recall@10 | 1.000 | 0.875 | -0.125 | fail |

The job failed with exit code 1, so the PR could not go green. The rules (`eval/gate.py`): fail if
`hybrid_rerank` loses more than 2 points of hit@8 or recall@10 overall, or more than 5 on any
question type. Retrieval is deterministic, so any drop is a real change on these questions.

The same gate on an unchanged branch reproduces every gated number exactly on GitHub's runner
(index rebuilt there from the `corpus-v1` release in 158 s).

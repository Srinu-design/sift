# Whole-corpus ingestion through Hopper

13th Gen Intel(R) Core(TM) i7-13700H, 20 threads; Docker engine 29.9.0; one sift-models (one model call at a time),
one sift-ingest, Hopper's 3 default workers x 20 slots for parse and activate,
N embed workers x 1 slot on the sift-embed queue.
The embedding cache is emptied before each run.

| Embed workers | Run | Seconds | Chunks/s | Embedded | Retried jobs | Failed jobs |
|---|---|---|---|---|---|---|
| 1 | 1 | 87.4 | 21.8 | 1007 | 0 | 0 |
| 2 | 1 | 85.6 | 22.2 | 1007 | 0 | 0 |
| 4 | 1 | 81.9 | 23.2 | 1007 | 0 | 0 |
| 1 | 2 | 87.6 | 21.7 | 1007 | 0 | 0 |
| 2 | 2 | 84.0 | 22.7 | 1007 | 0 | 0 |
| 4 | 2 | 80.9 | 23.5 | 1007 | 0 | 0 |

## What it shows

Mean of two runs: **87.5 s** with 1 embed worker, **84.8 s** with 2, **81.4 s** with 4. Four
times the workers buys 7%: ingestion is bound by sift-models, which runs one model call at a
time and already spreads that call over every core. Extra workers only overlap the database
work around each call. Every run embedded the same 1,007 unique texts (of 1,903 chunks: the
rest are identical across docs versions) and no job was retried or failed.

To ingest faster, add sift-models replicas (and as many `sift-embed` slots), not Hopper
workers. Two embed workers stay the default: one encoding, one ready, and the kill test still
has a survivor.

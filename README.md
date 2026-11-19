# certified-corruption-budget

Attach a **certified corruption budget** to leaderboard claims.

Every ranking claim gets a number `B_hat`: with probability at least `1 - alpha`,
at every time `t`, either the claim holds, or more than `B_hat` of the first
`t` records were corrupted. The guarantee holds even if an attacker watches
every published certificate and corrupts votes in bursts.

This is a product implementation of:

> Hamed Khosravi and Xiaoming Huo, *Certified Corruption Budgets: Anytime-Valid
> Leaderboard Claims under Adaptive Rigging*, arXiv:2610.10597, 2026.
> https://arxiv.org/abs/2610.10597

## Status

The product certifies one pair from a JSONL or CSV record file and prints a
human report or a `ccb/v1` JSON envelope. Edge and lead claims, insertion and
replacement costs, batch and streaming, and selection correction are
implemented and checked against the reference implementation's published
values. Leaderboard, replay, and the HTML demo are not built yet.

## What it will do

- Ingest pairwise records from JSONL or CSV.
- Emit edge and lead certificates with insertion or replacement costs.
- Emit simultaneous certificates for a whole leaderboard.
- Correct for selecting the best of `V` private model variants.
- Replay burst, flip, and spread attacks against the certificate and the
  paper's baselines.
- Produce a human table, stable JSON, and a static HTML report.

## CLI

Working now:

```sh
ccb certify --input records.jsonl --pair model-x,model-y --claim edge --cost replacement --alpha 0.05
ccb certify --input records.jsonl --pair model-x,model-y --json
ccb replay  --input null.jsonl --attack flip --budget 100 --replays 1000 --json
```

Planned:

```sh
ccb leaderboard --input records.jsonl --alpha 0.05
ccb demo        --input records.jsonl --out demo.html
```

## Input shape

One record per line:

```json
{"model_a":"model-x","model_b":"model-y","outcome":1.0,"t":0}
```

`outcome` is `1.0` if `model_a` wins, `0.0` if `model_b` wins, `0.5` for a tie.

## Licence

MIT is the recommended licence; the reference implementation is MIT. The owner
must confirm before a `LICENSE` file is added or anything is published.

## Not a claim of truth

A certificate is a statement about robustness to corrupted records, not proof
that a model is better. Read the output as: “the claim holds unless more than
`B_hat` records were corrupted.”

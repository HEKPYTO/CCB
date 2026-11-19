# Certified Corruption Budget

Attach a **certified corruption budget** to leaderboard claims.

Every ranking claim gets a number `B_hat`: with probability at least `1 - alpha`,
at every time `t`, either the claim holds, or more than `B_hat` of the first
`t` records were corrupted. The guarantee holds even if an attacker watches
every published certificate and corrupts votes in bursts.

This is a product implementation of:

> Hamed Khosravi and Xiaoming Huo, *Certified Corruption Budgets: Anytime-Valid
> Leaderboard Claims under Adaptive Rigging*, arXiv:2610.10597, 2026.
> https://arxiv.org/abs/2610.10597

This is an independent implementation. It is not affiliated with, endorsed by,
or reviewed by the paper's authors.

## Status

Local-first Python library and CLI. Edge and lead certificates, insertion and
replacement costs, batch and streaming, selection correction, closed-testing
and Bonferroni leaderboards, attack replays, a human table, a stable `ccb/v1`
JSON envelope, and a self-contained HTML report are implemented and tested.
Published on PyPI as `certified-corruption-budget`.

## Install

```sh
pip install certified-corruption-budget
```

Python 3.11 or newer; NumPy is the only runtime dependency.

## Example

```sh
$ ccb certify --input records.jsonl --pair model-x,model-y
pair          model-x > model-y
claim         edge
cost          replacement
alpha         0.05
B_rep         2
certified_at  13

At alpha=0.05, the edge claim that model-x has had an advantage over model-y holds unless more than 2 of the first 20 records were forged or altered (replacement cost).
```

## What it does

- Ingests pairwise records from JSONL or CSV.
- Emits edge and lead certificates with insertion or replacement costs.
- Emits simultaneous certificates for a whole leaderboard.
- Corrects for selecting the best of `V` private model variants.
- Replays burst, flip, and spread attacks against the certificate and the
  paper's baselines.
- Produces a human table, a stable JSON envelope, and a static HTML report.

## CLI

```sh
ccb certify     --input examples/records.jsonl --pair a,b --claim edge --cost replacement --alpha 0.05
ccb certify     --input examples/records.jsonl --pair a,b --json
ccb leaderboard --input examples/records.jsonl --method closed --cost replacement --alpha 0.05
ccb replay      --input examples/null.jsonl --attack flip --budget 100 --replays 1000 --json
ccb demo        --input examples/records.jsonl --out demo.html
ccb version
```

`ccb certify` and `ccb leaderboard` accept `--json` and print exactly one
`ccb/v1` envelope. Schemas live in `schema/` and ship inside the wheel.

## Input shape

One record per line:

```json
{"model_a":"model-x","model_b":"model-y","outcome":1.0,"t":0}
```

`outcome` is `1.0` if `model_a` wins, `0.0` if `model_b` wins, `0.5` for a tie.
`t` and `variant` are optional; `--format csv` reads the same columns from a
CSV header.

## Limitations

- A certificate is robustness to corrupted records, not proof that a model is
  better, and not fraud detection.
- `ccb replay` estimates false-statement rates by resampling the input stream;
  the rates are meaningful only when the uncorrupted claim is false.
- A leaderboard is a snapshot at the shortest input stream. There is no
  streaming update of a published leaderboard.
- A `-1` budget means no certificate was issued at the chosen `alpha`; it is
  not evidence against the claim.
- The guarantee assumes `alpha`, the horizon, the corruption cost, the pair
  set, and the variant count were fixed before the data was seen. Choices made
  after looking at the data need their own correction.
- Local files only. There is no hosted ingestion, account, or database.
- The paper is a preprint; the implementation is not peer reviewed.

## Security and contributing

Report vulnerabilities through a GitHub security advisory, not a public issue.

Development: `uv sync`, then `uv run pytest` and `uv run ruff check .`. A change
to a certificate or leaderboard formula needs a proof or a citation to the
paper; name the section in the function docstring.

## Licence

MIT (see `LICENSE`).

# ccb

Statistical core for certified corruption budgets.

| Module | Contents |
|---|---|
| `core.py` | 14-bet grid, insertion/replacement costs, wealth paths, deflation, budget bisection |
| `certificates.py` | `Certificate`, `certify_edge`, `certify_lead`, streaming `EdgeCertificate` and `LeadCertificate` |
| `records.py` | JSONL/CSV loading, validation, pair orientation, variant filter |
| `report.py` | claim statements and the `ccb/v1` envelope |
| `errors.py` | `CCBError` with the contract error code |
| `attacks.py` | burst, random burst, flip, spread, suppression |
| `baselines.py` | Wald, undeflated betting, per-step robust |
| `replay.py` | resampled false-statement replay, deterministic per seed |
| `leaderboard.py` | closed testing and Bonferroni over a pair family |
| `cli.py` | `ccb` entry point: `version`, `certify`, `replay`, `leaderboard`, `demo`; `V > 1` requires `--variant` |

Formulas follow arXiv:2610.10597 eq. (5)–(10) and Theorems 5.1/5.7. Values are
pinned against the reference implementation and `tests/golden/p70_1000_seed0.json`.

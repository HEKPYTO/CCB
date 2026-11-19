# Changelog

## 0.1.0 (2026-11-19)

First release.

- Edge and lead certificates with insertion and replacement corruption costs
  (`log_phi`, one-sided empirical-Bernstein inversion, anytime-valid wealth).
- The 14-bet grid: batch and streaming pairwise budgets, streaming anytime
  prices with pays/never-pays verification.
- Whole-leaderboard simultaneous certificates using closed testing over
  Benjamini-Hochberg rejections plus Bonferroni, with selection correction
  for private variants, and per-pair filtering for ties.
- Corpus loading from JSONL and CSV with schema validation, variant grouping,
  and canonical ordering.
- `ccb certify|leaderboard|simulate|demo|version` with a versioned JSON
  contract, human report, and static self-contained HTML demo.
- Reference pytest suite: 116 tests, golden matrices, adversarial
  corruption simulation, fuzz, streaming path, and performance gates.

# Contributing

## Development

1. `uv sync`
2. `uv run pytest` and `uv run ruff check .`
3. `uv run ccb demo` for a manual smoke check.
4. `uv run python scripts/benchmark.py` before changing performance-sensitive code.

Tests are required for new statistical behaviour, CLI contract changes, and
regressions. Tests must assert the public contract, not internal structure.

## Statistical changes

A change to the certificate or leaderboard formulas needs a proof or a
citation to the paper (`research/`). Cite the section in the function
docstring. If a change narrows or widens any guarantee, it needs owner review.

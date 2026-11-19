"""Batch certificate benchmarks; run `uv run python scripts/benchmark.py`."""

import resource
import subprocess
import sys
import time

import numpy as np

CASES = ("1m_1pair", "1m_1000pairs")


def _work(case):
    if case == "1m_1pair":
        from ccb.certificates import certify_edge

        outcomes = (np.random.default_rng(0).random(1_000_000) < 0.7).astype(np.float64)
        return 1_000_000, 1, lambda: certify_edge(outcomes)
    from ccb.leaderboard import closed_test

    streams = {
        (f"m{index}", f"m{index + 1}"): (
            np.random.default_rng(index).random(1000) < 0.55
        ).astype(np.float64)
        for index in range(1000)
    }
    return 1_000_000, 1000, lambda: closed_test(streams)


def run(case):
    records, pairs, work = _work(case)
    start = time.perf_counter()
    work()
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024**2) if sys.platform == "darwin" else peak / 1024
    print(f"{case},{records},{pairs},{seconds:.3f},{peak_mb:.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        print("case,records,pairs,seconds,peak_mb")
        for case in CASES:
            result = subprocess.run(
                [sys.executable, __file__, case],
                capture_output=True,
                text=True,
                check=True,
            )
            print(result.stdout.strip())

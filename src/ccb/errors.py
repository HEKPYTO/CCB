import numpy as np


class CCBError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validated_alpha(alpha) -> float:
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise CCBError("invalid_alpha", "alpha must be in (0, 1)")
    return float(alpha)


def validated_cost(cost: str) -> str:
    if cost not in ("insertion", "replacement"):
        raise CCBError("usage", "cost must be 'insertion' or 'replacement'")
    return cost


def validated_variants(selection_variants) -> int:
    if isinstance(selection_variants, bool) or not isinstance(
        selection_variants, (int, np.integer)
    ):
        raise TypeError("selection_variants must be a positive integer")
    if selection_variants < 1:
        raise ValueError("selection_variants must be a positive integer")
    return int(selection_variants)


def validated_outcomes(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.size and (
        not np.all(np.isfinite(result)) or result.min() < 0.0 or result.max() > 1.0
    ):
        raise CCBError("invalid_outcome", "outcome must be finite and in [0, 1]")
    return result

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import CCBError

FIELDS = ("model_a", "model_b", "outcome")
OPTIONAL = ("t", "variant")


@dataclass(frozen=True, slots=True)
class Record:
    model_a: str
    model_b: str
    outcome: float
    t: int | None = None
    variant: str | None = None


def _record(raw: dict, cache: dict) -> Record:
    missing = [field for field in FIELDS if field not in raw]
    if missing:
        raise CCBError("parse_error", f"missing field: {missing[0]}")
    extra = set(raw) - set(FIELDS) - set(OPTIONAL)
    if extra:
        raise CCBError("parse_error", f"unknown field: {min(extra)}")

    raw_names = [raw["model_a"], raw["model_b"]]
    if not all(isinstance(name, str) for name in raw_names):
        raise CCBError("parse_error", "model names must be strings")
    stripped = [name.strip() for name in raw_names]
    model_a, model_b = (cache.setdefault(name, name) for name in stripped)
    if not model_a or not model_b or model_a == model_b:
        raise CCBError("invalid_pair", "model names must be non-empty and distinct")

    outcome = raw["outcome"]
    if isinstance(outcome, bool) or not isinstance(outcome, (int, float)):
        raise CCBError("parse_error", "outcome must be a number")
    outcome = float(outcome)
    if not math.isfinite(outcome) or not 0.0 <= outcome <= 1.0:
        raise CCBError("invalid_outcome", "outcome must be finite and in [0, 1]")

    t = raw.get("t")
    if t is not None and (isinstance(t, bool) or not isinstance(t, int)):
        raise CCBError("parse_error", "t must be an integer")
    variant = raw.get("variant")
    if variant is not None:
        if not isinstance(variant, str):
            raise CCBError("parse_error", "variant must be a string")
        variant = variant.strip() or None

    return Record(model_a, model_b, outcome, t, variant)


def _from_csv(raw: dict) -> dict:
    row = {"model_a": raw["model_a"], "model_b": raw["model_b"]}
    try:
        row["outcome"] = float(raw["outcome"])
    except (TypeError, ValueError):
        raise CCBError("parse_error", "outcome must be a number") from None
    t = (raw.get("t") or "").strip()
    if t:
        try:
            row["t"] = int(t)
        except ValueError:
            raise CCBError("parse_error", "t must be an integer") from None
    variant = (raw.get("variant") or "").strip()
    if variant:
        row["variant"] = variant
    return row


def _read_jsonl(path: Path, cache: dict) -> list[Record]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CCBError("parse_error", f"invalid JSON: {exc.msg}") from None
            if not isinstance(raw, dict):
                raise CCBError("parse_error", "each line must be a JSON object")
            records.append(_record(raw, cache))
    return records


def _read_csv(path: Path, cache: dict) -> list[Record]:
    records = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [field for field in FIELDS if field not in header]
        if missing:
            raise CCBError("parse_error", f"missing column: {missing[0]}")
        extra = set(header) - set(FIELDS) - set(OPTIONAL)
        if extra:
            raise CCBError("parse_error", f"unknown column: {min(extra)}")
        for raw in reader:
            records.append(_record(_from_csv(raw), cache))
    return records


def load_records(path: str | Path, fmt: str | None = None) -> list[Record]:
    path = Path(path)
    fmt = fmt or {".jsonl": "jsonl", ".csv": "csv"}.get(path.suffix.lower())
    if fmt not in ("jsonl", "csv"):
        raise CCBError("unknown_format", "format must be jsonl or csv")
    cache: dict[str, str] = {}
    try:
        records = _read_csv(path, cache) if fmt == "csv" else _read_jsonl(path, cache)
    except UnicodeDecodeError:
        raise CCBError("parse_error", "input is not valid UTF-8") from None
    except csv.Error as exc:
        raise CCBError("parse_error", f"csv parse error: {exc}") from None
    except OSError as exc:
        raise CCBError("io_error", f"cannot read {path}: {exc.strerror}") from None
    if not records:
        raise CCBError("empty_input", "no records")
    timed = [record.t is not None for record in records]
    if any(timed) and not all(timed):
        raise CCBError("mixed_time", "some records have t and some do not")
    if all(timed):
        records.sort(key=lambda record: record.t)
    return records


def pair_streams(
    records: list[Record],
    *,
    variant: str | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    records = list(records)
    if not records:
        raise CCBError("empty_input", "no records")
    if variant is not None:
        records = [record for record in records if record.variant == variant]
        if not records:
            raise CCBError("empty_input", f"no records for variant {variant}")
    elif len({record.variant for record in records if record.variant}) > 1:
        raise CCBError("mixed_variants", "multiple variants present; pass variant")

    streams: dict[tuple[str, str], list[float]] = {}
    for record in records:
        key = (record.model_a, record.model_b)
        reverse = (record.model_b, record.model_a)
        if key in streams:
            streams[key].append(record.outcome)
        elif reverse in streams:
            streams[reverse].append(1.0 - record.outcome)
        else:
            streams[key] = [record.outcome]
    return {key: np.array(values) for key, values in streams.items()}

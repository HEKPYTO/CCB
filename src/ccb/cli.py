import argparse
import sys
from pathlib import Path

from . import __version__
from .certificates import Certificate, certify_edge, certify_lead
from .errors import CCBError
from .records import load_records, pair_streams
from .replay import DEFAULT_BUDGETS, ReplayResult, replay
from .report import envelope, statement


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _budget_list(text: str) -> list[int]:
    values = [_non_negative(part) for part in text.split(",") if part]
    if not values:
        raise argparse.ArgumentTypeError("at least one budget is required")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccb",
        description="Certified corruption budgets for leaderboard claims.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="print the version")
    certify = sub.add_parser("certify", help="certify one pair")
    certify.add_argument("--input", required=True)
    certify.add_argument("--format", choices=("jsonl", "csv"))
    certify.add_argument("--pair")
    certify.add_argument("--claim", choices=("edge", "lead"), default="edge")
    certify.add_argument(
        "--cost", choices=("insertion", "replacement"), default="replacement"
    )
    certify.add_argument("--alpha", type=float, default=0.05)
    certify.add_argument("--variant")
    certify.add_argument("--selection-variants", type=int, default=1)
    certify.add_argument("--reverse-pair", action="store_true")
    certify.add_argument("--json", action="store_true")
    certify.add_argument("--out")
    certify.add_argument("--force", action="store_true")
    replay_cmd = sub.add_parser(
        "replay", help="replay an attack against the certificate"
    )
    replay_cmd.add_argument("--input", required=True)
    replay_cmd.add_argument("--format", choices=("jsonl", "csv"))
    replay_cmd.add_argument(
        "--attack",
        choices=("burst", "burst_random", "flip", "spread"),
        default="burst",
    )
    replay_cmd.add_argument("--budget", type=_non_negative)
    replay_cmd.add_argument("--budgets", type=_budget_list)
    replay_cmd.add_argument("--replays", type=_positive, default=1000)
    replay_cmd.add_argument("--seed", type=_non_negative, default=0)
    replay_cmd.add_argument("--alpha", type=float, default=0.05)
    replay_cmd.add_argument("--json", action="store_true")
    replay_cmd.add_argument("--out")
    replay_cmd.add_argument("--force", action="store_true")
    return parser


def _select(streams, pair: str | None):
    if pair is None:
        if len(streams) != 1:
            raise CCBError("multiple_pairs", "input has multiple pairs; pass --pair")
        (model_a, model_b), outcomes = next(iter(streams.items()))
        return model_a, model_b, outcomes
    model_a, _, model_b = pair.partition(",")
    model_a, model_b = model_a.strip(), model_b.strip()
    if (model_a, model_b) in streams:
        return model_a, model_b, streams[(model_a, model_b)]
    if (model_b, model_a) in streams:
        return model_a, model_b, 1.0 - streams[(model_b, model_a)]
    raise CCBError("invalid_pair", f"pair {model_a},{model_b} not found in input")


def _certify_command(args: argparse.Namespace) -> tuple[str, str, Certificate]:
    records = load_records(args.input, args.format)
    streams = pair_streams(records, variant=args.variant)
    model_a, model_b, outcomes = _select(streams, args.pair)
    if args.reverse_pair:
        model_a, model_b = model_b, model_a
        outcomes = 1.0 - outcomes
    if args.claim == "lead":
        certificate = certify_lead(
            outcomes,
            alpha=args.alpha,
            cost=args.cost,
            selection_variants=args.selection_variants,
        )
    else:
        certificate = certify_edge(
            outcomes,
            alpha=args.alpha,
            cost=args.cost,
            selection_variants=args.selection_variants,
        )
    return model_a, model_b, certificate


def _payload(model_a: str, model_b: str, certificate: Certificate) -> dict:
    return {
        "pair": [model_a, model_b],
        "statement": statement(certificate, model_a, model_b),
        "certificate": {
            "claim": certificate.claim,
            "cost": certificate.cost,
            "alpha": certificate.alpha,
            "n": certificate.n,
            "tolerance": certificate.tolerance,
            "certified_at": certificate.certified_at,
            "selection_variants": certificate.selection_variants,
        },
    }


def _render(args: argparse.Namespace, payload: dict) -> str:
    if args.json:
        return envelope(args.command, payload)
    certificate = payload["certificate"]
    label = "B_ins" if certificate["cost"] == "insertion" else "B_rep"
    model_a, model_b = payload["pair"]
    return "\n".join(
        [
            f"{'pair':<14}{model_a} > {model_b}",
            f"{'claim':<14}{certificate['claim']}",
            f"{'cost':<14}{certificate['cost']}",
            f"{'alpha':<14}{certificate['alpha']:g}",
            f"{label:<14}{certificate['tolerance']}",
            f"{'certified_at':<14}{certificate['certified_at']}",
            "",
            payload["statement"],
        ]
    )


def _replay_command(args: argparse.Namespace) -> str:
    records = load_records(args.input, args.format)
    streams = pair_streams(records)
    if len(streams) != 1:
        raise CCBError("multiple_pairs", "replay needs a single pair")
    stream = next(iter(streams.values()))
    budgets = [args.budget] if args.budget is not None else args.budgets
    result = replay(
        stream,
        attack=args.attack,
        budgets=budgets if budgets is not None else DEFAULT_BUDGETS,
        replays=args.replays,
        seed=args.seed,
        alpha=args.alpha,
    )
    if args.json:
        return envelope("replay", _replay_payload(result))
    return _replay_table(result)


def _replay_payload(result: ReplayResult) -> dict:
    return {
        "attack": result.attack,
        "records": result.records,
        "replays": result.replays,
        "seed": result.seed,
        "alpha": result.alpha,
        "budgets": result.budgets,
        "methods": result.methods,
        "rates": {
            method: {str(budget): rate for budget, rate in values.items()}
            for method, values in result.rates.items()
        },
        "se": {
            method: {str(budget): error for budget, error in values.items()}
            for method, values in result.se.items()
        },
    }


def _replay_table(result: ReplayResult) -> str:
    header = f"{'method':<14}" + "".join(f"B={budget:<9}" for budget in result.budgets)
    rows = [header]
    for method, values in result.rates.items():
        rows.append(
            f"{method:<14}"
            + "".join(f"{values[budget]:<11.4f}" for budget in result.budgets)
        )
    meta = [
        f"{'attack':<14}{result.attack}",
        f"{'records':<14}{result.records}",
        f"{'replays':<14}{result.replays}",
        f"{'seed':<14}{result.seed}",
        f"{'alpha':<14}{result.alpha:g}",
    ]
    caution = (
        "Rates estimate the false-statement probability only when the "
        "uncorrupted claim is false."
    )
    return "\n".join([*meta, "", *rows, "", caution])


def _write(text: str, args: argparse.Namespace) -> None:
    if not args.out:
        print(text)
        return
    path = Path(args.out)
    if path.exists() and not args.force:
        raise CCBError("output_exists", f"{path} exists; pass --force")
    try:
        path.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        raise CCBError("io_error", f"cannot write {path}: {exc.strerror}") from None


def _fail(args: argparse.Namespace, error: CCBError) -> None:
    payload = {"code": error.code, "message": error.message}
    if getattr(args, "json", False):
        print(envelope(args.command, None, payload))
    else:
        print(f"error: {error.code}: {error.message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "certify" and args.selection_variants > 1 and not args.variant:
        parser.error("--selection-variants > 1 requires --variant")
    if (
        args.command == "replay"
        and args.budget is not None
        and args.budgets is not None
    ):
        parser.error("use one of --budget or --budgets")
    if args.command == "version":
        print(f"ccb {__version__}")
        return 0
    try:
        if args.command == "replay":
            _write(_replay_command(args), args)
        else:
            model_a, model_b, certificate = _certify_command(args)
            _write(_render(args, _payload(model_a, model_b, certificate)), args)
    except CCBError as error:
        _fail(args, error)
        return 1
    return 0

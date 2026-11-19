import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .certificates import certify_edge, certify_lead
from .errors import CCBError
from .leaderboard import bonferroni, closed_test
from .records import load_records, pair_streams
from .replay import DEFAULT_BUDGETS, replay
from .report import (
    certify_payload,
    certify_text,
    demo_html,
    envelope,
    leaderboard_payload,
    leaderboard_text,
    replay_text,
)


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

    board = sub.add_parser("leaderboard", help="certify a family of pairs")
    board.add_argument("--input", required=True)
    board.add_argument("--format", choices=("jsonl", "csv"))
    board.add_argument("--method", choices=("closed", "bonferroni"), default="closed")
    board.add_argument(
        "--cost", choices=("insertion", "replacement"), default="replacement"
    )
    board.add_argument("--alpha", type=float, default=0.05)
    board.add_argument("--both-directions", action="store_true")
    board.add_argument("--json", action="store_true")
    board.add_argument("--out")
    board.add_argument("--force", action="store_true")

    demo = sub.add_parser("demo", help="write a self-contained HTML report")
    demo.add_argument("--input", required=True)
    demo.add_argument("--format", choices=("jsonl", "csv"))
    demo.add_argument("--out", default="demo.html")
    demo.add_argument("--seed", type=_non_negative, default=0)
    demo.add_argument("--force", action="store_true")
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


def _certify_command(args: argparse.Namespace) -> str:
    records = load_records(args.input, args.format)
    streams = pair_streams(records, variant=args.variant)
    model_a, model_b, outcomes = _select(streams, args.pair)
    if args.reverse_pair:
        model_a, model_b = model_b, model_a
        outcomes = 1.0 - outcomes
    certify = certify_lead if args.claim == "lead" else certify_edge
    certificate = certify(
        outcomes,
        alpha=args.alpha,
        cost=args.cost,
        selection_variants=args.selection_variants,
    )
    payload = certify_payload(model_a, model_b, certificate)
    return envelope("certify", payload) if args.json else certify_text(payload)


def _replay_command(args: argparse.Namespace) -> str:
    records = load_records(args.input, args.format)
    streams = pair_streams(records)
    if len(streams) != 1:
        raise CCBError("multiple_pairs", "replay needs a single pair")
    budgets = [args.budget] if args.budget is not None else args.budgets
    result = replay(
        next(iter(streams.values())),
        attack=args.attack,
        budgets=budgets if budgets is not None else DEFAULT_BUDGETS,
        replays=args.replays,
        seed=args.seed,
        alpha=args.alpha,
    )
    payload = result.to_dict()
    return envelope("replay", payload) if args.json else replay_text(payload)


def _leaderboard_command(args: argparse.Namespace) -> str:
    records = load_records(args.input, args.format)
    certify = closed_test if args.method == "closed" else bonferroni
    results = certify(
        pair_streams(records),
        alpha=args.alpha,
        cost=args.cost,
        both_directions=args.both_directions,
    )
    payload = leaderboard_payload(results, args.method, args.cost, args.alpha)
    return envelope("leaderboard", payload) if args.json else leaderboard_text(payload)


def _demo_command(args: argparse.Namespace) -> str:
    path = Path(args.input)
    records = load_records(path, args.format)
    streams = pair_streams(records)
    results = closed_test(streams)
    horizon = min(values.size for values in streams.values())
    null = (np.random.default_rng(args.seed).random(min(horizon, 1000)) < 0.5).astype(
        float
    )
    simulation = replay(
        null,
        attack="flip",
        budgets=DEFAULT_BUDGETS,
        replays=100,
        seed=args.seed,
    ).to_dict()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return demo_html(streams, results, simulation, digest)


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


_COMMANDS = {
    "certify": _certify_command,
    "replay": _replay_command,
    "leaderboard": _leaderboard_command,
    "demo": _demo_command,
}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "certify" and args.selection_variants < 1:
        parser.error("--selection-variants must be positive")
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
        _write(_COMMANDS[args.command](args), args)
        if args.command == "demo":
            print(f"wrote {args.out}")
    except CCBError as error:
        _fail(args, error)
        return 1
    return 0

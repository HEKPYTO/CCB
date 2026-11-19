import html
import json

from .certificates import Certificate

SCHEMA = "ccb/v1"


def statement(certificate: Certificate, model_a: str, model_b: str) -> str:
    alpha = f"{certificate.alpha:g}"
    n = certificate.n
    if certificate.tolerance < 0:
        return (
            f"At alpha={alpha}, no {certificate.claim} claim is certified for "
            f"{model_a} over {model_b} on the first {n} records."
        )
    action = "forged" if certificate.cost == "insertion" else "forged or altered"
    if certificate.claim == "edge":
        return (
            f"At alpha={alpha}, the edge claim that {model_a} has had an "
            f"advantage over {model_b} holds unless more than "
            f"{certificate.tolerance} of the first {n} records were {action} "
            f"({certificate.cost} cost)."
        )
    return (
        f"At alpha={alpha}, the lead claim that {model_a} leads {model_b} on "
        f"the uncorrupted margin over the first {n} records holds unless more "
        f"than {certificate.tolerance} of those records were {action} "
        f"({certificate.cost} cost)."
    )


def envelope(command: str, data, error=None) -> str:
    return json.dumps(
        {"schema": SCHEMA, "command": command, "data": data, "error": error}
    )


def certificate_dict(certificate: Certificate) -> dict:
    return {
        "claim": certificate.claim,
        "cost": certificate.cost,
        "alpha": certificate.alpha,
        "n": certificate.n,
        "tolerance": certificate.tolerance,
        "certified_at": certificate.certified_at,
        "selection_variants": certificate.selection_variants,
    }


def certify_payload(model_a: str, model_b: str, certificate: Certificate) -> dict:
    return {
        "pair": [model_a, model_b],
        "statement": statement(certificate, model_a, model_b),
        "certificate": certificate_dict(certificate),
    }


def certify_text(payload: dict) -> str:
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


def replay_text(data: dict) -> str:
    budgets = data["budgets"]
    header = f"{'method':<14}" + "".join(f"B={budget:<9}" for budget in budgets)
    rows = [header]
    for method, values in data["rates"].items():
        rows.append(
            f"{method:<14}"
            + "".join(f"{values[str(budget)]:<11.4f}" for budget in budgets)
        )
    meta = [
        f"{'attack':<14}{data['attack']}",
        f"{'records':<14}{data['records']}",
        f"{'replays':<14}{data['replays']}",
        f"{'seed':<14}{data['seed']}",
        f"{'alpha':<14}{data['alpha']:g}",
    ]
    caution = (
        "Rates estimate the false-statement probability only when the "
        "uncorrupted claim is false."
    )
    return "\n".join([*meta, "", *rows, "", caution])


def leaderboard_payload(results, method: str, cost: str, alpha: float) -> dict:
    return {
        "method": method,
        "cost": cost,
        "alpha": alpha,
        "claims": [
            {
                "model_a": result.model_a,
                "model_b": result.model_b,
                "n": result.n,
                "tolerance": result.tolerance,
                "certified_at": result.certified_at,
                "status": result.status,
            }
            for result in results
        ],
    }


def leaderboard_text(data: dict) -> str:
    label = "B_ins" if data["cost"] == "insertion" else "B_rep"
    rows = [f"{'pair':<28}{'n':>8}  {label:>6}  status"]
    for row in data["claims"]:
        pair = f"{row['model_a']} > {row['model_b']}"
        rows.append(f"{pair:<28}{row['n']:>8}  {row['tolerance']:>6}  {row['status']}")
    meta = [
        f"{'method':<14}{data['method']}",
        f"{'cost':<14}{data['cost']}",
        f"{'alpha':<14}{data['alpha']:g}",
        "",
    ]
    return "\n".join([*meta, *rows])


_METHOD_LABELS = {
    "wald": "Wald",
    "naive": "Undeflated",
    "per_step": "Per-step",
    "insertion": "Insertion",
    "replacement": "Replacement",
}

_CSS = """
body { font-family: Georgia, serif; margin: 2rem auto; max-width: 760px; color: #1b1b1b; }
h1 { font-size: 1.5rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { border-bottom: 1px solid #d8d8d8; padding: 0.35rem 0.5rem; text-align: left; }
code { font-family: monospace; font-size: 0.8rem; }
.muted { color: #666; font-size: 0.85rem; }
"""


def _rate_table(data: dict) -> str:
    budgets = data["budgets"]
    header = "".join(f"<th>B={budget}</th>" for budget in budgets)
    rows = []
    for method, values in data["rates"].items():
        cells = "".join(f"<td>{values[str(budget)]:.3f}</td>" for budget in budgets)
        rows.append(f"<tr><td>{_METHOD_LABELS[method]}</td>{cells}</tr>")
    return (
        "<table><thead><tr><th>method</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _rate_chart(data: dict) -> str:
    budget = data["budgets"][-1]
    bars = []
    for index, (method, values) in enumerate(data["rates"].items()):
        rate = values[str(budget)]
        y = 8 + index * 26
        width = max(round(rate * 400), 1)
        bars.append(
            f'<rect x="120" y="{y}" width="{width}" height="18" fill="#1f6feb"/>'
        )
        bars.append(
            f'<text x="4" y="{y + 13}" font-size="12">'
            f"{html.escape(_METHOD_LABELS[method])}</text>"
        )
        bars.append(
            f'<text x="{128 + width}" y="{y + 13}" font-size="12">{rate:.3f}</text>'
        )
    height = 8 + len(data["rates"]) * 26
    return (
        f'<svg width="560" height="{height}" role="img" '
        f'aria-label="False-statement rates at B={budget}">' + "".join(bars) + "</svg>"
    )


def demo_html(streams, results, replay_data: dict, digest: str) -> str:
    records = sum(values.size for values in streams.values())
    certified = sum(1 for result in results if result.tolerance >= 0)
    rows = []
    for result in results:
        pair = f"{html.escape(result.model_a)} &gt; {html.escape(result.model_b)}"
        ratio = (
            f"{result.tolerance / result.n:.3f}"
            if result.tolerance >= 0 and result.n
            else "n/a"
        )
        rows.append(
            f"<tr><td>{pair}</td><td>{result.n}</td>"
            f"<td>{result.tolerance}</td><td>{ratio}</td>"
            f"<td>{result.certified_at}</td></tr>"
        )
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Certified corruption budgets</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<h1>Certified corruption budgets</h1>",
        (
            '<p class="muted">Hamed Khosravi and Xiaoming Huo, '
            "<em>Certified Corruption Budgets: Anytime-Valid Leaderboard "
            "Claims under Adaptive Rigging</em>, arXiv:2610.10597.</p>"
        ),
        (
            f"<p>alpha {results[0].alpha:g} &middot; cost {results[0].cost} "
            f"&middot; method {results[0].method}</p>"
        ),
        f'<p class="muted">input sha256 <code>{digest}</code></p>',
        "<h2>Summary</h2>",
        (
            f"<p>{len(streams)} pairs &middot; {records} records &middot; "
            f"{certified} certified</p>"
        ),
        "<h2>Pairs</h2>",
        (
            "<table><thead><tr><th>pair</th><th>n</th><th>B_hat</th>"
            "<th>B_hat/n</th><th>certified_at</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        ),
        f"<h2>Replay: {html.escape(replay_data['attack'])} attack on a simulated null leaderboard</h2>",
        (
            f'<p class="muted">{replay_data["replays"]} replays, seed '
            f"{replay_data['seed']}, {replay_data['records']} fair-coin records. "
            "This panel demonstrates the method; it is not a claim about the "
            "input data.</p>"
        ),
        _rate_table(replay_data),
        (
            f'<p class="muted">False-statement rates at '
            f"B={replay_data['budgets'][-1]}.</p>"
        ),
        f"<p>{_rate_chart(replay_data)}</p>",
        (
            '<p class="muted">A certificate is not proof of truth; it is '
            "robustness to corrupted records.</p>"
        ),
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)

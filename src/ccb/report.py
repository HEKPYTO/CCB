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

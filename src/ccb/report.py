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

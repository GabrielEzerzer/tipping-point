# The market-latent template with every placeholder filled, for tests that need a file
# the strict validator and the freeze accept.

import hashlib
import itertools
import re
from pathlib import Path

from tipping_point.ews import prereg

TEMPLATE = Path(__file__).resolve().parents[4] / "paper" / "prereg" / "TEMPLATE-market-latent.toml"


def filled(text: str, baseline_sha256: str = "0" * 64) -> str:
    count, scaling = itertools.count(), itertools.count()
    per_symbol = len(prereg.LATENT_PRIMARIES + prereg.LATENT_SPARES)

    def digest(match: re.Match[str]) -> str:
        if match[1] == "scaling_sha256":
            seed = f"scaling{next(scaling) // per_symbol}"
        else:
            seed = f"{match[1]}{next(count)}"
        return f'{match[1]} = "{hashlib.sha256(seed.encode()).hexdigest()}"'

    text = text.replace('id = "<lowercase-slug>"', 'id = "market-latent"', 1)
    text = re.sub(r'^(\w+_sha256) = "<[^"]*>"', digest, text, flags=re.M)
    text = re.sub(
        r'^run_id = "<[^"]*>"', lambda _: f'run_id = "run-{next(count)}"', text, flags=re.M
    )
    text = re.sub(r'^data_commit = "<[^"]*>"', 'data_commit = "abc1234"', text, flags=re.M)
    text = re.sub(r'^(sha256 = )"<[^"]*>"', rf'\1"{baseline_sha256}"', text, flags=re.M)
    text = re.sub(r'^judged = "<[^"]*>"', 'judged = "same-asset"', text, flags=re.M)
    text = re.sub(r'^against = "<[^"]*>"', 'against = "each"', text, flags=re.M)
    text = re.sub(r'^tolerance = "<[^"]*>"', "tolerance = 0.02", text, flags=re.M)
    return re.sub(r'"<[^"]*>"', '"set"', text)


def on_baseline(ews: bytes) -> str:
    return filled(TEMPLATE.read_text("utf-8"), hashlib.sha256(ews).hexdigest())

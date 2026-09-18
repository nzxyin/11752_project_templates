"""Text-frontend registry + the loader used by TTSDataModule, scripts/synthesize.py,
and scripts/evaluate.py to reconstruct whichever frontend a given
`preprocessed_dir` was built with (recorded in `preprocess_config.json` +
`symbols.json` by scripts/preprocess.py) -- single source of truth so training
and inference/evaluation never drift apart.
"""
from __future__ import annotations

import json
import os

from src.data.frontends.base import TextFrontend
from src.data.frontends.g2p_en_frontend import G2pEnFrontend
from src.data.frontends.phonemizer_frontend import PhonemizerFrontend

FRONTENDS: dict[str, type[TextFrontend]] = {
    "g2p_en": G2pEnFrontend,
    "phonemizer": PhonemizerFrontend,
}


def load_frontend_for_preprocessed_dir(preprocessed_dir: str) -> TextFrontend:
    with open(os.path.join(preprocessed_dir, "preprocess_config.json")) as f:
        config = json.load(f)
    with open(os.path.join(preprocessed_dir, "symbols.json")) as f:
        symbols = json.load(f)

    name = config["frontend"]
    kwargs = {"language": config["language"]} if name == "phonemizer" else {}
    return FRONTENDS[name](symbols, **kwargs)

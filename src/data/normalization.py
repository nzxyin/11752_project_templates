"""Optional text-normalization pre-pass, applied to raw text before it reaches
a text frontend (src/data/frontends/). NOT NEEDED FOR LJSPEECH: its
metadata.csv already ships pre-normalized transcripts (numbers/abbreviations
already expanded), so scripts/preprocess.py's default `--normalize none` is
correct and sufficient here -- this hook exists purely as an extension point
for pointing the pipeline at a messier, non-LJSpeech dataset later.

The "nemo" backend requires the `normalize` extra (`uv sync --extra normalize`)
AND is Linux/WSL/conda-forge only -- its core dependency `pynini` has no
native Windows wheels. See README.md's Setup section.
"""
from __future__ import annotations

_NEMO_NORMALIZER = None


def normalize_text(text: str, backend: str = "none") -> str:
    if backend == "none":
        return text
    if backend == "nemo":
        global _NEMO_NORMALIZER
        if _NEMO_NORMALIZER is None:
            from nemo_text_processing.text_normalization.normalize import Normalizer

            _NEMO_NORMALIZER = Normalizer(input_case="cased", lang="en")
        return _NEMO_NORMALIZER.normalize(text, verbose=False, punct_post_process=True)
    raise ValueError(f"Unknown normalization backend {backend!r}, expected 'none' or 'nemo'")

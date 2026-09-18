"""ARPAbet phoneme frontend for English, via `g2p_en` (CMUdict-backed G2P).
No forced alignment or external tool needed. This is the default frontend --
zero system dependencies beyond the `preprocess`/`synthesize` extras.

Filelist entries store the phoneme sequence in curly braces, space-separated,
e.g. "{HH AH0 L OW1} {W ER1 L D}" for "Hello world" -- the ming024/FastSpeech2
convention, which lets punctuation appear un-braced between/after phoneme groups.
"""
from __future__ import annotations

import re

from src.data.frontends.base import TextFrontend

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"

# CMUdict / ARPAbet phoneme set (39 phonemes, stress-marked variants for vowels).
_vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
_consonants = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R",
    "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
]
_arpabet = [f"@{p}{stress}" for p in _vowels for stress in ("0", "1", "2")] + [f"@{p}" for p in _consonants]

SYMBOLS = [_pad] + list(_special) + list(_punctuation) + _arpabet

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

_G2P = None


class G2pEnFrontend(TextFrontend):
    name = "g2p_en"

    def __init__(self, symbols: list[str] | None = None):
        # symbols is accepted (and ignored beyond a sanity check) for interface
        # uniformity with PhonemizerFrontend -- this vocab is fixed, not data-driven.
        self.symbols = symbols if symbols is not None else list(SYMBOLS)
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}

    def encode_for_storage(self, raw_text: str) -> str:
        return g2p_phone_string(raw_text)

    def stored_string_to_sequence(self, stored: str) -> list[int]:
        return phones_to_sequence(stored, self.symbol_to_id)


def phones_to_sequence(phoneme_string: str, symbol_to_id: dict[str, int] | None = None) -> list[int]:
    """"{HH AH0 L OW1} {W ER1 L D}!" -> [id, id, ..., id]"""
    symbol_to_id = symbol_to_id if symbol_to_id is not None else {s: i for i, s in enumerate(SYMBOLS)}
    sequence: list[int] = []
    text = phoneme_string
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(text, symbol_to_id)
            break
        pre, phonemes, text = m.group(1), m.group(2), m.group(3)
        sequence += _symbols_to_sequence(pre, symbol_to_id)
        sequence += _symbols_to_sequence([f"@{p}" for p in phonemes.split()], symbol_to_id)
    return sequence


def _symbols_to_sequence(symbols, symbol_to_id: dict[str, int]) -> list[int]:
    return [symbol_to_id[s] for s in symbols if s in symbol_to_id]


def g2p_phone_string(text: str) -> str:
    """raw text -> "{PH ON EME} {W ER1 D} ..." (this module's curly-brace-per-word
    convention, see module docstring), using g2p_en -- no forced alignment
    needed. Shared by scripts/preprocess.py and scripts/synthesize.py so
    training and inference phonemize identically."""
    global _G2P
    if _G2P is None:
        _ensure_nltk_data()
        from g2p_en import G2p

        _G2P = G2p()

    groups: list[str] = []
    current: list[str] = []
    for p in _G2P(text):
        if p == " ":
            if current:
                groups.append("{" + " ".join(current) + "}")
                current = []
        elif p.isalnum():  # ARPAbet phoneme, e.g. "AH0", "K"
            current.append(p)
        else:  # punctuation -- passed through bare, outside any {..} group
            if current:
                groups.append("{" + " ".join(current) + "}")
                current = []
            groups.append(p)
    if current:
        groups.append("{" + " ".join(current) + "}")
    return " ".join(groups)


def _ensure_nltk_data():
    """g2p_en's own startup check only auto-downloads the (now-renamed) old
    'averaged_perceptron_tagger' resource; recent nltk actually looks up
    'averaged_perceptron_tagger_eng' at run time, which g2p_en never fetches --
    so a fresh environment fails on first use with a LookupError deep inside
    nltk.pos_tag. Fetch everything g2p_en needs up front instead."""
    import nltk

    for pkg in ("cmudict", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:  # offline, or nltk's download server unreachable
            print(f"[WARNING] nltk.download({pkg!r}) failed ({e}); g2p_en may error below.")

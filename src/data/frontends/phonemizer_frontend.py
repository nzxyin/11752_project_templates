"""IPA phoneme frontend via `phonemizer` and the espeak-ng backend, the
convention many non-English-only or Matcha-TTS-style implementations expect,
as an alternative to the default ARPAbet `g2p_en` frontend.

Requires the `phonemizer` extra (`uv sync --extra phonemizer`) and the
espeak-ng system binary (not pip-installable). See README.md's Setup section
for per-OS install instructions.

Unlike ARPAbet's fixed 39-phoneme set, espeak's IPA output varies by language
and is not enumerated here, so this frontend tokenizes at the character level
and builds its vocabulary from the training corpus during preprocessing
(`PhonemizerFrontend.build_vocab`), persisted to `symbols.json`.
"""
from __future__ import annotations

from src.data.frontends.base import TextFrontend

_PAD = "_"


class PhonemizerFrontend(TextFrontend):
    name = "phonemizer"

    def __init__(self, symbols: list[str] | None, language: str = "en-us"):
        # symbols is None during scripts/preprocess.py's first pass, before the
        # corpus has been scanned to build the vocab (see build_vocab below).
        # encode_for_storage doesn't need it; only stored_string_to_sequence does.
        self.symbols = symbols or []
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.language = language
        self._backend = None  # lazy EspeakBackend, mirrors g2p_en_frontend's lazy _G2P

    def _ensure_backend(self):
        if self._backend is None:
            from phonemizer.backend import EspeakBackend
            from phonemizer.separator import Separator

            self._backend = EspeakBackend(self.language, preserve_punctuation=True, with_stress=True)
            # Explicit separator (no "|") so phonemizer's own formatting can
            # never collide with the filelist's "|"-delimited field separator.
            self._separator = Separator(word=" ", syllable="", phone="")

    def encode_for_storage(self, raw_text: str) -> str:
        self._ensure_backend()
        # phonemize() takes/returns lists; a single utterance in, single string out.
        (ipa,) = self._backend.phonemize([raw_text], separator=self._separator, strip=True)
        # Defensive: filelist lines are "basename|speaker|stored|raw_text".
        # Guard against a stray "|" or newline regardless of separator config.
        return ipa.replace("|", " ").replace("\n", " ")

    def stored_string_to_sequence(self, stored: str) -> list[int]:
        return [self.symbol_to_id[c] for c in stored if c in self.symbol_to_id]

    @staticmethod
    def build_vocab(stored_strings: list[str]) -> list[str]:
        """Scan every utterance's encode_for_storage() output once (done by
        scripts/preprocess.py during the corpus pass) and build a deterministic
        character-level vocab, pad symbol first at index 0."""
        chars = sorted({c for s in stored_strings for c in s})
        return [_PAD] + chars

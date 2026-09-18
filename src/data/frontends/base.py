"""The text-frontend plug-in contract: raw text -> token ids, plus the on-disk
serialization used by filelists (`scripts/preprocess.py`'s `train.txt`/`val.txt`).
See `g2p_en_frontend.py` (ARPAbet, via g2p_en) and `phonemizer_frontend.py`
(IPA, via phonemizer+espeak-ng) for the two concrete implementations, and
`__init__.py` for the registry/loader used by the datamodule, synthesize.py,
and evaluate.py.
"""
from __future__ import annotations

import abc


class TextFrontend(abc.ABC):
    name: str
    symbols: list[str]  # index 0 is reserved as the pad symbol

    @abc.abstractmethod
    def encode_for_storage(self, raw_text: str) -> str:
        """raw text -> a string safe to store as a filelist's phoneme column
        (must not contain "|" or a newline). Frontend-owned serialization,
        opaque to callers, e.g. g2p_en's curly-brace-per-word convention, or
        phonemizer's raw IPA string."""

    @abc.abstractmethod
    def stored_string_to_sequence(self, stored: str) -> list[int]:
        """Inverse of encode_for_storage, using self.symbols. This is what
        src.data.dataset.TTSDataset calls per item at load time."""

    def encode(self, raw_text: str) -> list[int]:
        """Convenience for inference: raw text -> token ids directly, skipping
        the round-trip through a stored string."""
        return self.stored_string_to_sequence(self.encode_for_storage(raw_text))

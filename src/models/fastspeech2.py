"""PLACEHOLDER for a FastSpeech2 implementation (Ren et al., ICLR 2021,
https://arxiv.org/abs/2006.04558). Currently just embedding + a tiny
Transformer encoder + naive length-matched linear mel projection -- exists
ONLY to make `experiment=debug` and scripts/synthesize.py|evaluate.py runnable
end-to-end out of the box as an integration smoke test.

This is NOT a real FastSpeech2: there is no variance adaptor (duration/pitch/
energy predictors), no length regulator, and no aligner -- just a fixed
frames-per-phoneme heuristic used at inference to decide how long the output
should be, and per-sample linear interpolation (not length regulation) to
stretch the encoder output to that length during training too. Replace this
class (and `network` in configs/model/fastspeech2.yaml) with a real
implementation satisfying src.models.base.BaseTTSModel -- see README.md's
"The model contract" section, and this README's "Implementing FastSpeech2"
section for the shape of what's missing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseTTSModel
from src.utils.tools import get_mask_from_lengths


class FastSpeech2Placeholder(BaseTTSModel):
    def __init__(
        self,
        n_symbols: int,
        n_mel_channels: int,
        n_speakers: int = 1,
        multi_speaker: bool = False,
        hidden: int = 256,
        encoder_layers: int = 2,
        encoder_heads: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        inference_length_multiplier: float = 8.0,
    ):
        super().__init__()
        self.multi_speaker = multi_speaker
        self.inference_length_multiplier = inference_length_multiplier

        self.embedding = nn.Embedding(n_symbols, hidden, padding_idx=0)
        self.speaker_embedding = nn.Embedding(n_speakers, hidden) if multi_speaker else None
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, encoder_heads, ffn_dim, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.mel_proj = nn.Linear(hidden, n_mel_channels)

    def _encode(self, text_ids: torch.LongTensor, src_lens: torch.LongTensor, speakers=None) -> torch.Tensor:
        pad_mask = get_mask_from_lengths(src_lens, text_ids.shape[1])
        x = self.embedding(text_ids)
        if self.speaker_embedding is not None and speakers is not None:
            x = x + self.speaker_embedding(speakers).unsqueeze(1)
        return self.encoder(x, src_key_padding_mask=pad_mask)

    @staticmethod
    def _length_match(enc: torch.Tensor, src_lens: torch.Tensor, target_lens: torch.Tensor, max_target_len: int) -> torch.Tensor:
        """Per-sample linear interpolation along time -- a naive placeholder for
        real duration prediction / length regulation. (B, T_text, H) -> (B, max_target_len, H)."""
        out = enc.new_zeros(enc.shape[0], max_target_len, enc.shape[-1])
        for i in range(enc.shape[0]):
            src = enc[i : i + 1, : src_lens[i]].transpose(1, 2)
            tgt_len = max(1, int(target_lens[i]))
            resized = F.interpolate(src, size=tgt_len, mode="linear", align_corners=False)
            out[i, :tgt_len] = resized.transpose(1, 2)[0]
        return out

    def forward(self, batch: dict) -> dict:
        enc = self._encode(batch["texts"], batch["src_lens"], batch.get("speakers"))
        matched = self._length_match(enc, batch["src_lens"], batch["mel_lens"], batch["max_mel_len"])
        mel_pred = self.mel_proj(matched)

        pad_mask = get_mask_from_lengths(batch["mel_lens"], batch["max_mel_len"])
        keep_mask = (~pad_mask).unsqueeze(-1)
        loss = F.l1_loss(mel_pred * keep_mask, batch["mels"] * keep_mask, reduction="sum") / keep_mask.sum().clamp(min=1)

        return {"loss": loss, "mel_pred": mel_pred}

    def synthesize(self, text_ids: torch.LongTensor, src_lens: torch.LongTensor, max_src_len: int | None = None, **kwargs) -> dict:
        speakers = kwargs.get("speaker_id")
        enc = self._encode(text_ids, src_lens, speakers)

        target_lens = kwargs.get("target_mel_lens")
        if target_lens is None:
            target_lens = (src_lens.float() * self.inference_length_multiplier).round().long().clamp(min=1)

        matched = self._length_match(enc, src_lens, target_lens, int(target_lens.max().item()))
        mel = self.mel_proj(matched)
        return {"mel": mel, "mel_lens": target_lens}

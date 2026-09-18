"""Thin wrapper around the vendored BigVGAN generator (src/vocoders/bigvgan/,
see its THIRD_PARTY_NOTICES.md) for loading NVIDIA's official pretrained
checkpoints from the Hugging Face Hub and running mel -> waveform.

Requires the `vocoder` extra (`uv sync --extra vocoder`) for `huggingface_hub`.
"""
from __future__ import annotations

import torch

from src.vocoders.bigvgan import BigVGAN


class BigVGANVocoder:
    """Feed it mels produced by `src.vocoders.mel.bigvgan_mel_spectrogram` (or an
    acoustic model trained to reconstruct those) -- other mel conventions will
    produce degraded or wrong audio, BigVGAN was trained on this one specifically.
    """

    def __init__(
        self,
        checkpoint: str = "nvidia/bigvgan_v2_22khz_80band_256x",
        device: str = "cpu",
        use_cuda_kernel: bool = False,
    ):
        self.model = BigVGAN.from_pretrained(checkpoint, use_cuda_kernel=use_cuda_kernel)
        self.model.remove_weight_norm()
        self.model.eval().to(device)
        self.device = device
        self.sampling_rate = self.model.h["sampling_rate"]
        self.hop_size = self.model.h["hop_size"]
        self.num_mels = self.model.h["num_mels"]

    @torch.inference_mode()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, num_mels, T) or (num_mels, T) log-mel. Returns (B, T_wav) (or
        (T_wav,) for a 2D input) waveform in [-1, 1]."""
        squeeze = mel.dim() == 2
        if squeeze:
            mel = mel.unsqueeze(0)
        wav = self.model(mel.to(self.device)).squeeze(1)
        return wav.squeeze(0) if squeeze else wav

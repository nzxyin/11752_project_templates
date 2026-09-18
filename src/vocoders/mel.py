"""Mel-spectrogram extraction, reproducing bit-for-bit the convention BigVGAN's
official pretrained checkpoints were trained on. This is what makes a mel
*predicted* by an acoustic model -- trained to reconstruct ground-truth mels
extracted the same way (see scripts/preprocess.py) -- actually sound right once
handed to the vocoder.

Ported from NVIDIA/BigVGAN's meldataset.py::mel_spectrogram (MIT license, see
src/vocoders/bigvgan/THIRD_PARTY_NOTICES.md) -- manual reflect-pad +
center=False torch.stft, librosa "slaney" mel filterbank, natural log with a
1e-5 clamp.

Defaults match the 22050 Hz / 80-mel / n_fft=1024 / hop=256 / win=1024 recipe
of `nvidia/bigvgan_v2_22khz_80band_256x` -- see configs/data/ljspeech.yaml.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

_mel_basis_cache: dict = {}
_hann_window_cache: dict = {}


def _stft_magnitude(y: torch.Tensor, n_fft: int, hop_size: int, win_size: int) -> torch.Tensor:
    """y: (B, T) waveform in [-1, 1]. Returns (B, n_fft // 2 + 1, T_mel)."""
    key = f"{n_fft}_{win_size}_{y.device}"
    if key not in _hann_window_cache:
        _hann_window_cache[key] = torch.hann_window(win_size, device=y.device)
    hann_window = _hann_window_cache[key]

    pad = (n_fft - hop_size) // 2
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)


def bigvgan_mel_spectrogram(
    y: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 80,
    sampling_rate: int = 22050,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: int = 0,
    fmax: int | None = None,
) -> torch.Tensor:
    """y: (B, T) waveform in [-1, 1]. Returns (B, num_mels, T_mel), natural-log mel."""
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{y.device}"
    if key not in _mel_basis_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[key] = torch.from_numpy(mel).float().to(y.device)
    mel_basis = _mel_basis_cache[key]

    spec = _stft_magnitude(y, n_fft, hop_size, win_size)
    mel_spec = torch.matmul(mel_basis, spec)
    return torch.log(torch.clamp(mel_spec, min=1e-5))

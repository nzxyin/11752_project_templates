"""Mel -> waveform vocoding for turning a predicted mel-spectrogram into audio.
BigVGAN-only (nvidia/bigvgan_v2_22khz_80band_256x by default), matching
LJSpeech's native 22050 Hz and src/vocoders/mel.py's mel convention.

Loaded lazily (only when actually requested) so `import src.vocoders` never
requires the `vocoder` extra to be installed.
"""
from __future__ import annotations

DEFAULT_CHECKPOINT = "nvidia/bigvgan_v2_22khz_80band_256x"


def load_vocoder(checkpoint: str | None = None, device: str = "cpu"):
    """checkpoint: a Hugging Face Hub repo id, or a local directory containing
    config.json + bigvgan_generator.pt. Defaults to BigVGAN's official
    22050 Hz/80-mel checkpoint. Returns a callable
    `vocoder(mel: (B, num_mels, T)) -> (B, T_wav)`."""
    from src.vocoders.bigvgan_vocoder import BigVGANVocoder

    return BigVGANVocoder(checkpoint or DEFAULT_CHECKPOINT, device=device)

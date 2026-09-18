"""The plug-in contract a concrete acoustic model (FastSpeech2, Matcha-TTS,
F5-TTS, etc.) must satisfy to work with `src.lightning_module.TTSLightningModule`,
`src.data.dataset.TTSDataset`/`TTSDataModule`, and `scripts/synthesize.py` /
`scripts/evaluate.py`. It is deliberately silent on duration modeling, decoder
architecture (regression, diffusion, or flow-matching), and the number of
internal stages, all of which are the subclass's business. See
`src/models/example.py` for a minimal concrete implementation, and README.md's
"The model contract" section for a walkthrough of adding a new model.
"""
from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseTTSModel(nn.Module, abc.ABC):
    requires_reference_audio: bool = False
    """Set True on a subclass whose `synthesize` needs an audio prompt (e.g.
    F5-TTS-style zero-shot voice cloning). `scripts/synthesize.py` then requires
    `--ref_audio`/`--ref_text` and forwards them as `ref_mel`/`ref_text_ids`
    kwargs. Otherwise those flags are unused."""

    @abc.abstractmethod
    def forward(self, batch: dict) -> dict:
        """Teacher-forced training/validation step.

        `batch` is whatever `src.data.dataset.collate_fn` produces:
            texts: (B, T_text) LongTensor, token ids from the active text frontend
            src_lens: (B,) LongTensor
            max_src_len: int
            mels: (B, T_mel, n_mel) FloatTensor, ground truth
            mel_lens: (B,) LongTensor
            max_mel_len: int
            speakers: (B,) LongTensor, speaker id (0 for single-speaker data)
            ids, raw_texts: python lists (not tensors)

        Must return a dict containing:
            "loss": 0-d tensor, backpropagated directly by
                TTSLightningModule.training_step/validation_step.
        May include any other scalar entry (0-d tensor, python int/float),
        auto-logged as f"{split}/{key}". By convention, also include:
            "mel_pred": (B, T_mel, n_mel) FloatTensor. Enables the default
                ground-truth-vs-predicted mel image logged every validation
                epoch (TTSLightningModule._log_mel_example). This is optional;
                a diffusion/flow model for which a denoised sample is not a
                free byproduct of the training step may omit it and rely on
                `scripts/evaluate.py` for real quality inspection instead.
        """

    @abc.abstractmethod
    def synthesize(
        self,
        text_ids: torch.LongTensor,
        src_lens: torch.LongTensor,
        max_src_len: int | None = None,
        **kwargs,
    ) -> dict:
        """Inference: text to mel, with no ground-truth mel available.

        kwargs are open-ended and model-specific (e.g. Matcha-TTS's ODE step
        count/temperature, F5-TTS's `ref_mel`/`ref_text_ids`/`cfg_scale`).
        `scripts/synthesize.py --model_kwarg key=value` forwards arbitrary
        entries here.

        Must return a dict containing:
            "mel": (B, T_mel, n_mel) FloatTensor.
        May include "mel_lens", "alignment", etc.; callers ignore unused keys.
        """

    def log_artifacts(self, batch: dict, outputs: dict) -> dict:
        """Optional extra validation figures beyond the default GT-vs-predicted
        mel pair (e.g. an attention/alignment plot). Return
        {tag: matplotlib.figure.Figure}."""
        return {}

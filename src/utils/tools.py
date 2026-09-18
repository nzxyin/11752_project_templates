"""Small tensor / plotting utilities shared across the data pipeline and model."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import torch


def get_mask_from_lengths(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """(B,) lengths -> (B, max_len) boolean mask, True where PADDED."""
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1)
    return mask


def pad_1d(inputs: list[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    max_len = max(x.shape[0] for x in inputs)
    return np.stack(
        [np.pad(x, (0, max_len - x.shape[0]), mode="constant", constant_values=pad_value) for x in inputs]
    )


def pad_2d(inputs: list[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    """Pad a list of (T_i, C) arrays -> (B, max_T, C)."""
    max_len = max(x.shape[0] for x in inputs)
    out = np.full((len(inputs), max_len, inputs[0].shape[1]), pad_value, dtype=np.float32)
    for i, x in enumerate(inputs):
        out[i, : x.shape[0], :] = x
    return out


def pad_tensor_2d(inputs: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Pad a list of (T_i, C) tensors -> (B, max_T, C)."""
    max_len = max(x.shape[0] for x in inputs)
    out = inputs[0].new_full((len(inputs), max_len, inputs[0].shape[1]), pad_value)
    for i, x in enumerate(inputs):
        out[i, : x.shape[0], :] = x
    return out


def plot_mel(data: list[np.ndarray], titles: list[str] | None = None):
    """Plot one or more (n_mel, T) mel-spectrograms side by side, for TB/W&B image logging."""
    fig, axes = plt.subplots(len(data), 1, squeeze=False, figsize=(10, 2.5 * len(data)))
    if titles is None:
        titles = [None] * len(data)
    for i, mel in enumerate(data):
        ax = axes[i][0]
        im = ax.imshow(mel, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
        if titles[i] is not None:
            ax.set_title(titles[i], fontsize="medium")
        ax.set_anchor("W")
    fig.tight_layout()
    return fig

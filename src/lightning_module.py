"""Generic PyTorch Lightning wrapper around any src.models.base.BaseTTSModel
subclass. It handles training/validation steps, loss-agnostic scalar logging,
AdamW(fused)-plus-Hydra-instantiated-scheduler optimization, and periodic
mel-spectrogram image logging (TensorBoard and/or W&B, whichever loggers are
attached to the Trainer). This module never needs to change when swapping in a
different acoustic model. See configs/model/example.yaml for how the actual
model is nested under `network:` and instantiated via Hydra, and
src/models/base.py for the contract a model must satisfy.

DDP / multi-node: nothing model-specific is needed here. See
configs/trainer/ddp.yaml and the README for how that is driven from the
Trainer side. `sync_dist=True` below makes the logged val losses correct when
running under multiple ranks.
"""
from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import LightningModule

from src.models.base import BaseTTSModel
from src.utils.pylogger import get_pylogger
from src.utils.tools import plot_mel

log = get_pylogger(__name__)


def _plain(x):
    # Store hparams as plain python dict/list/str/... rather than OmegaConf
    # DictConfig: torch>=2.6 defaults `torch.load(weights_only=True)`, which
    # refuses to unpickle DictConfig, breaking `load_from_checkpoint` unless
    # every OmegaConf type is explicitly allowlisted. Plain containers avoid
    # the problem entirely and are what get saved into ckpt["hyper_parameters"].
    return OmegaConf.to_container(x, resolve=True) if isinstance(x, (DictConfig, ListConfig)) else x


def _cfg(x):
    # Inverse of _plain(), for building submodules: `network` etc. arrive as
    # DictConfig when constructed fresh from Hydra, but as a plain dict when
    # reconstructed by `load_from_checkpoint` (which replays the plain hparams
    # saved above). Normalize to DictConfig either way so `.attr` access works.
    return OmegaConf.create(x) if isinstance(x, dict) else x


def _is_loggable_scalar(v) -> bool:
    return isinstance(v, (int, float)) or (torch.is_tensor(v) and v.dim() == 0)


class TTSLightningModule(LightningModule):
    def __init__(self, network: DictConfig, optimizer: DictConfig, scheduler: DictConfig, **extra_kwargs):
        super().__init__()

        network = _cfg(network)

        # Stash everything so ckpt.hparams is enough to rebuild the model without
        # the original hydra config. The datamodule fills in extra_kwargs
        # (e.g. n_symbols) at runtime, see TTSDataModule.stats. extra_kwargs
        # is spread into the top-level dict, not nested under an "extra_kwargs"
        # key, so load_from_checkpoint's `cls(**hparams)` reconstruction passes
        # e.g. n_symbols=81 back in as a real keyword, not a mis-named container.
        self.save_hyperparameters(
            dict(
                network=_plain(network),
                optimizer=_plain(optimizer),
                scheduler=_plain(scheduler),
                **{k: _plain(v) for k, v in extra_kwargs.items()},
            ),
            logger=False,
        )

        self.model: BaseTTSModel = hydra.utils.instantiate(network, _recursive_=False, **extra_kwargs)

    def forward(self, batch: dict) -> dict:
        return self.model(batch)

    def training_step(self, batch: dict, batch_idx: int):
        outputs = self(batch)
        batch_size = batch["texts"].shape[0]
        self.log_dict(
            {f"train/{k}": v for k, v in outputs.items() if _is_loggable_scalar(v)},
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True, batch_size=batch_size)
        return outputs["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        outputs = self(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in outputs.items() if _is_loggable_scalar(v)},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["texts"].shape[0],
        )

        if batch_idx == 0:
            self._log_mel_example(batch, outputs)

        return outputs["loss"]

    def _log_mel_example(self, batch: dict, outputs: dict):
        """Log a ground-truth vs. predicted mel-spectrogram pair (when the
        model's forward() included "mel_pred"), plus any model-specific extra
        figures (src.models.base.BaseTTSModel.log_artifacts), to whichever
        experiment tracker(s) are attached."""
        figures = {}

        mel_pred = outputs.get("mel_pred")
        if mel_pred is not None:
            idx = 0
            mel_len = int(batch["mel_lens"][idx].item())
            gt_mel = batch["mels"][idx, :mel_len].transpose(0, 1).detach().cpu().numpy()
            pred_mel = mel_pred[idx, :mel_len].transpose(0, 1).detach().cpu().numpy()
            figures["val/mel_example"] = plot_mel([gt_mel, pred_mel], ["Ground-Truth Mel", "Predicted Mel"])

        figures.update(self.model.log_artifacts(batch, outputs) or {})

        if not figures:
            return

        for logger in self._loggers_list():
            for tag, fig in figures.items():
                if logger.__class__.__name__ == "TensorBoardLogger":
                    logger.experiment.add_figure(tag, fig, global_step=self.global_step)
                elif logger.__class__.__name__ == "WandbLogger":
                    import wandb

                    logger.experiment.log({tag: wandb.Image(fig)}, step=self.global_step)

        import matplotlib.pyplot as plt

        for fig in figures.values():
            plt.close(fig)

    def _loggers_list(self):
        if self.logger is None:
            return []
        # Trainer(logger=[...]) exposes LoggerCollection-like access via self.loggers (PL >= 1.9)
        return getattr(self, "loggers", [self.logger])

    def configure_optimizers(self):
        # self.hparams.{optimizer,scheduler} are plain dicts (see __init__).
        # Wrap back into a DictConfig so hydra.utils.instantiate can resolve `_target_`.
        optimizer_cfg = OmegaConf.create(self.hparams.optimizer)
        # `fused=True` AdamW requires all params to be CUDA tensors. Fall back to the
        # unfused kernel automatically so the same config also works for the CPU debug
        # run (configs/experiment/debug.yaml) instead of erroring.
        if optimizer_cfg.get("fused", False) and not all(p.is_cuda for p in self.parameters()):
            log.warning("optimizer.fused=true requested but model is not on CUDA. Disabling fused AdamW.")
            optimizer_cfg.fused = False
        optimizer = hydra.utils.instantiate(optimizer_cfg, params=self.parameters())

        scheduler = hydra.utils.instantiate(OmegaConf.create(self.hparams.scheduler), optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

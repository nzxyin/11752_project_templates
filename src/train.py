"""Hydra + PyTorch Lightning entry point.

Examples:
    python src/train.py
    python src/train.py trainer.max_steps=50000 data.batch_size=32
    python src/train.py logger=wandb
    python src/train.py logger=both                     # TensorBoard + W&B together
    python src/train.py experiment=debug                 # tiny CPU smoke test
    python src/train.py train=false test=true ckpt_path=logs/runs/.../checkpoints/last.ckpt
"""
from __future__ import annotations

import pathlib

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# Hydra normally resolves a relative config_path against the calling module's
# __file__. That detection uses sys.modules['__main__'], which for the
# installed `tts-train` console script (see [project.scripts] in
# pyproject.toml) is setuptools' tiny generated launcher, not this file. That
# breaks relative resolution (misread as a config *package* to import rather
# than a directory) for the entry point, while `python -m src.train` and
# `python src/train.py` both still work fine. An absolute path sidesteps the
# ambiguity entirely and works identically for every invocation style.
_CONFIG_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "configs")


def _instantiate_callbacks(cfg: DictConfig) -> list:
    callbacks = []
    if not cfg:
        return callbacks
    for name, cb_cfg in cfg.items():
        if cb_cfg is not None and "_target_" in cb_cfg:
            log.info(f"Instantiating callback <{cb_cfg._target_}> ({name})")
            callbacks.append(hydra.utils.instantiate(cb_cfg))
    return callbacks


def _instantiate_loggers(cfg: DictConfig) -> list:
    loggers = []
    if not cfg:
        return loggers
    for name, lg_cfg in cfg.items():
        if lg_cfg is not None and "_target_" in lg_cfg:
            log.info(f"Instantiating logger <{lg_cfg._target_}> ({name})")
            loggers.append(hydra.utils.instantiate(lg_cfg))
    return loggers


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    # Extension point (e.g. n_symbols, computed from the persisted text-frontend
    # vocab). See TTSDataModule.stats. Requires scripts/preprocess.py to have
    # already run against data.preprocessed_dir.
    stats = datamodule.stats

    log.info(f"Instantiating model <{cfg.model._target_}>")
    # _recursive_=False: cfg.model.optimizer / cfg.model.scheduler / cfg.model.network
    # carry their own _target_ but must not be eagerly instantiated here.
    # TTSLightningModule instantiates the network itself, and the optimizer once
    # model.parameters() exists.
    model = hydra.utils.instantiate(cfg.model, _recursive_=False, **stats)

    log.info("Instantiating callbacks")
    callbacks = _instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers")
    loggers = _instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if cfg.get("train", True):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test", False):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path if cfg.get("train", True) else cfg.get("ckpt_path")
        if not ckpt_path:
            log.warning("No checkpoint found, using current model weights for testing.")
            ckpt_path = None
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

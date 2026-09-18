"""LightningDataModule around TTSDataset. Optionally buckets utterances by mel
length before batching (data.sort=true) to cut padding waste, similar to the
ming024/FastSpeech2 reference implementation's `batch_size * group_size` scheme.
"""
from __future__ import annotations

import os
import random

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Sampler

from src.data.dataset import TTSDataset, collate_fn
from src.data.frontends import load_frontend_for_preprocessed_dir


class BucketBatchSampler(Sampler):
    """Groups nearby-length utterances into the same batch: chunk the (optionally
    shuffled) dataset into windows of `batch_size * batch_group_size`, sort each
    window by mel length, slice into batches, then shuffle batch order."""

    def __init__(self, dataset: TTSDataset, batch_size: int, batch_group_size: int, drop_last: bool, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.window = max(batch_size * batch_group_size, batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.mel_lens = None  # populated lazily -- see __iter__

    def _lengths(self) -> np.ndarray:
        if self.mel_lens is None:
            self.mel_lens = np.array(
                [np.load(os.path.join(self.dataset.preprocessed_dir, "mel", f"{s}-mel-{b}.npy")).shape[0]
                 for b, s in zip(self.dataset.basenames, self.dataset.speakers)]
            )
        return self.mel_lens

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        lens = self._lengths()
        batches = []
        for start in range(0, len(indices), self.window):
            chunk = indices[start : start + self.window]
            chunk.sort(key=lambda i: lens[i])
            for b_start in range(0, len(chunk), self.batch_size):
                batch = chunk[b_start : b_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


class TTSDataModule(LightningDataModule):
    def __init__(
        self,
        preprocessed_dir: str,
        train_filelist: str = "train.txt",
        val_filelist: str = "val.txt",
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        sort: bool = True,
        drop_last: bool = True,
        batch_group_size: int = 4,
        n_mel_channels: int = 80,
        multi_speaker: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_set: TTSDataset | None = None
        self.val_set: TTSDataset | None = None
        self._frontend = None

    def setup(self, stage: str | None = None):
        if self._frontend is None:
            self._frontend = load_frontend_for_preprocessed_dir(self.hparams.preprocessed_dir)
        if self.train_set is None:
            self.train_set = TTSDataset(self.hparams.train_filelist, self.hparams.preprocessed_dir, self._frontend)
        if self.val_set is None:
            self.val_set = TTSDataset(self.hparams.val_filelist, self.hparams.preprocessed_dir, self._frontend)

    def train_dataloader(self) -> DataLoader:
        if self.hparams.sort:
            sampler = BucketBatchSampler(
                self.train_set,
                self.hparams.batch_size,
                self.hparams.batch_group_size,
                self.hparams.drop_last,
                shuffle=True,
            )
            return DataLoader(
                self.train_set,
                batch_sampler=sampler,
                collate_fn=collate_fn,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=self.hparams.drop_last,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    @property
    def stats(self) -> dict:
        """Extension point forwarded into hydra.utils.instantiate(cfg.model, **stats)
        in src/train.py. Always includes n_symbols (from the persisted text
        frontend vocab, since that can't be known until preprocessing has run).
        Override/extend this if a real model needs something else computed
        from the actual preprocessed data at construction time."""
        if self._frontend is None:
            self._frontend = load_frontend_for_preprocessed_dir(self.hparams.preprocessed_dir)
        return {"n_symbols": len(self._frontend.symbols)}

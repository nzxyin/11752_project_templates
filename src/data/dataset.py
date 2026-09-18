"""Dataset over preprocessed LJSpeech features produced by scripts/preprocess.py.

Filelist format (one line per utterance, produced by preprocessing):
    basename|speaker|{frontend-specific phoneme/token string}|raw_text

Per-utterance mel is expected under `preprocessed_dir` as:
    mel/{speaker}-mel-{basename}.npy    (T_mel, n_mel_channels)
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.frontends.base import TextFrontend


class TTSDataset(Dataset):
    def __init__(
        self,
        filelist_name: str,
        preprocessed_dir: str,
        frontend: TextFrontend,
        sort: bool = False,
        drop_last: bool = False,
    ):
        self.preprocessed_dir = preprocessed_dir
        self.frontend = frontend
        self.sort = sort
        self.drop_last = drop_last

        self.basenames, self.speakers, self.phones, self.raw_texts = self._load_filelist(
            os.path.join(preprocessed_dir, filelist_name)
        )

        speaker_map_path = os.path.join(preprocessed_dir, "speakers.json")
        if os.path.isfile(speaker_map_path):
            import json

            with open(speaker_map_path) as f:
                self.speaker_map = json.load(f)
        else:
            self.speaker_map = {"LJSpeech": 0}

    @staticmethod
    def _load_filelist(path: str):
        basenames, speakers, phones, raw_texts = [], [], [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                basename, speaker, phone, raw_text = line.split("|")
                basenames.append(basename)
                speakers.append(speaker)
                phones.append(phone)
                raw_texts.append(raw_text)
        return basenames, speakers, phones, raw_texts

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx: int) -> dict:
        basename = self.basenames[idx]
        speaker = self.speakers[idx]
        speaker_id = self.speaker_map.get(speaker, 0)
        phone_ids = np.array(self.frontend.stored_string_to_sequence(self.phones[idx]))

        mel = np.load(os.path.join(self.preprocessed_dir, "mel", f"{speaker}-mel-{basename}.npy"))

        return {
            "id": basename,
            "speaker_id": speaker_id,
            "text": phone_ids,
            "raw_text": self.raw_texts[idx],
            "mel": mel,
        }


def collate_fn(batch: list[dict]) -> dict:
    ids = [b["id"] for b in batch]
    raw_texts = [b["raw_text"] for b in batch]

    src_lens = torch.LongTensor([len(b["text"]) for b in batch])
    mel_lens = torch.LongTensor([b["mel"].shape[0] for b in batch])

    max_src_len = int(src_lens.max().item())
    max_mel_len = int(mel_lens.max().item())

    texts = torch.LongTensor(_pad_1d([b["text"] for b in batch], max_src_len))
    mels = torch.FloatTensor(_pad_2d([b["mel"] for b in batch], max_mel_len))
    speakers = torch.LongTensor([b["speaker_id"] for b in batch])

    return {
        "ids": ids,
        "raw_texts": raw_texts,
        "texts": texts,
        "src_lens": src_lens,
        "max_src_len": max_src_len,
        "mels": mels,
        "mel_lens": mel_lens,
        "max_mel_len": max_mel_len,
        "speakers": speakers,
    }


def _pad_1d(seqs: list[np.ndarray], max_len: int) -> np.ndarray:
    out = np.zeros((len(seqs), max_len), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out


def _pad_2d(seqs: list[np.ndarray], max_len: int) -> np.ndarray:
    out = np.zeros((len(seqs), max_len, seqs[0].shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0], :] = s
    return out

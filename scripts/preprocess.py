"""Preprocess LJSpeech (kept at its NATIVE 22050 Hz, with no resampling) into
the features a generic acoustic model trains on: mel-spectrograms (matching
the BigVGANv2 22kHz vocoder's convention) and phoneme/token sequences from a
pluggable text frontend, plus the train/val filelists this template's
src/data/dataset.py consumes.

Text frontend
-------------
`--frontend {g2p_en,phonemizer}` (default g2p_en) picks how raw text becomes
token ids (src/data/frontends/). g2p_en needs no system dependencies beyond
the `preprocess` extra; phonemizer needs the `phonemizer` extra and the
espeak-ng system binary (see README.md's Setup section). `--language` only
applies to the phonemizer frontend (default "en-us").

`--normalize {none,nemo}` (default none) optionally runs a text-normalization
pass before phonemization. This is not needed for LJSpeech, whose metadata.csv
transcripts are already normalized; it is an extension point for other,
messier datasets. The "nemo" backend needs the `normalize` extra and is
Linux/WSL/conda-forge only, since its `pynini` dependency has no Windows wheels.

Writes, under `preprocessed_dir` (configs/paths/default.yaml -> paths.preprocessed_dir):
    mel/{speaker}-mel-{basename}.npy
    train.txt, val.txt, speakers.json, symbols.json, stats.json, preprocess_config.json

Usage:
    python scripts/preprocess.py --ljspeech_dir data/raw/LJSpeech-1.1 --out_dir data/preprocessed
    python scripts/preprocess.py --frontend phonemizer --language en-us --ljspeech_dir data/raw/LJSpeech-1.1 --out_dir data/preprocessed
"""
from __future__ import annotations

import argparse
import json
import os
import random

import librosa
import numpy as np
import torch
from tqdm import tqdm

from src.data.frontends import FRONTENDS
from src.data.frontends.phonemizer_frontend import PhonemizerFrontend
from src.data.normalization import normalize_text
from src.vocoders.mel import bigvgan_mel_spectrogram

SAMPLING_RATE = 22050   # LJSpeech's native rate, no resampling needed
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MEL_CHANNELS = 80     # matches nvidia/bigvgan_v2_22khz_80band_256x
MEL_FMIN = 0
MEL_FMAX = None         # Nyquist, matches that checkpoint's config (fmax=null)
SILENCE_TRIM_DB = 30    # librosa.effects.trim threshold for leading/trailing silence


def _mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """(T_mel, n_mel) log-mel spectrogram, matching the BigVGANv2 22kHz convention."""
    y = torch.from_numpy(wav).unsqueeze(0)
    mel = bigvgan_mel_spectrogram(y, N_FFT, N_MEL_CHANNELS, SAMPLING_RATE, HOP_LENGTH, WIN_LENGTH, MEL_FMIN, MEL_FMAX)
    return mel.squeeze(0).transpose(0, 1).numpy()  # (T_mel, n_mel)


def process_utterance(basename: str, speaker: str, raw_text: str, ljspeech_dir: str, out_dir: str, frontend, normalize_backend: str):
    wav_path = os.path.join(ljspeech_dir, "wavs", f"{basename}.wav")
    wav, _ = librosa.load(wav_path, sr=SAMPLING_RATE)
    wav, _ = librosa.effects.trim(wav, top_db=SILENCE_TRIM_DB)
    wav = wav.astype(np.float32)
    if len(wav) < HOP_LENGTH * 4:
        return None

    mel = _mel_spectrogram(wav)

    text = normalize_text(raw_text, normalize_backend)
    stored = frontend.encode_for_storage(text)
    if not stored.strip():
        return None

    np.save(os.path.join(out_dir, "mel", f"{speaker}-mel-{basename}.npy"), mel)

    # (sum, sum-of-squares, count) rather than the mel array itself. This keeps
    # memory flat across a full-corpus run instead of holding every utterance's
    # mel in RAM simultaneously just to compute stats.json at the end.
    mel_stats = (float(mel.sum()), float((mel.astype(np.float64) ** 2).sum()), mel.size)
    return stored, mel_stats


def main(args):
    os.makedirs(os.path.join(args.out_dir, "mel"), exist_ok=True)

    frontend_cls = FRONTENDS[args.frontend]
    frontend = frontend_cls(None, language=args.language) if args.frontend == "phonemizer" else frontend_cls()

    speaker = "LJSpeech"
    metadata_path = os.path.join(args.ljspeech_dir, "metadata.csv")
    entries = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                entries.append((parts[0], parts[2]))

    results = []
    mel_sum, mel_sumsq, mel_count = 0.0, 0.0, 0
    for basename, raw_text in tqdm(entries, desc=f"Preprocessing LJSpeech (--frontend {args.frontend})"):
        result = process_utterance(basename, speaker, raw_text, args.ljspeech_dir, args.out_dir, frontend, args.normalize)
        if result is None:
            continue
        stored, (s, ssq, n) = result
        results.append((basename, stored, raw_text))
        mel_sum += s
        mel_sumsq += ssq
        mel_count += n

    if args.frontend == "phonemizer":
        # espeak's IPA output is not a fixed enumerable set. Build the vocab
        # from what was actually observed in this corpus, then re-encode.
        frontend.symbols = PhonemizerFrontend.build_vocab([r[1] for r in results])
        frontend.symbol_to_id = {s: i for i, s in enumerate(frontend.symbols)}

    lines = [f"{basename}|{speaker}|{stored}|{raw_text}" for basename, stored, raw_text in results]

    with open(os.path.join(args.out_dir, "symbols.json"), "w") as f:
        json.dump(frontend.symbols, f, indent=2)

    with open(os.path.join(args.out_dir, "speakers.json"), "w") as f:
        json.dump({speaker: 0}, f, indent=2)

    mel_mean = mel_sum / mel_count if mel_count else 0.0
    mel_var = mel_sumsq / mel_count - mel_mean**2 if mel_count else 0.0
    stats = {
        "mel_mean": mel_mean,
        "mel_std": max(mel_var, 0.0) ** 0.5,
        "n_utterances": len(lines),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Records exactly how these features were extracted, so scripts/synthesize.py
    # and scripts/evaluate.py can reconstruct a matching frontend/vocoder without
    # the user having to remember and keep CLI flags in sync by hand.
    with open(os.path.join(args.out_dir, "preprocess_config.json"), "w") as f:
        json.dump(
            {
                "frontend": args.frontend,
                "language": args.language,
                "normalize": args.normalize,
                "sampling_rate": SAMPLING_RATE,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "win_length": WIN_LENGTH,
                "n_mel_channels": N_MEL_CHANNELS,
                "fmin": MEL_FMIN,
                "fmax": MEL_FMAX,
            },
            f,
            indent=2,
        )

    random.seed(args.seed)
    random.shuffle(lines)
    val_lines, train_lines = lines[: args.val_size], lines[args.val_size :]
    with open(os.path.join(args.out_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(args.out_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"Done. {len(train_lines)} train / {len(val_lines)} val utterances written to {args.out_dir}")
    print(f"Frontend: {args.frontend} ({len(frontend.symbols)} symbols), see {args.out_dir}/symbols.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=list(FRONTENDS.keys()), default="g2p_en", help="text -> token frontend, see src/data/frontends/")
    parser.add_argument("--language", default="en-us", help="--frontend phonemizer only")
    parser.add_argument("--normalize", choices=["none", "nemo"], default="none", help="optional text-normalization pre-pass; not needed for LJSpeech")
    parser.add_argument("--ljspeech_dir", default="data/raw/LJSpeech-1.1")
    parser.add_argument("--out_dir", default="data/preprocessed")
    parser.add_argument("--val_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def cli():
    """Entry point for the `tts-preprocess` uv tool (see [project.scripts]
    in pyproject.toml). Equivalent to `uv run python scripts/preprocess.py ...`."""
    main(build_parser().parse_args())


if __name__ == "__main__":
    cli()

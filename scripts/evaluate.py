"""Objective evaluation: synthesize a filelist's utterances, vocode them, and
score against the original LJSpeech ground-truth wavs with the VERSA toolkit
(https://github.com/wavlab-speech/versa). Requires the `evaluate` extra
(`uv sync --extra evaluate`) and the `vocoder` extra.

Since scripts/preprocess.py keeps LJSpeech at its NATIVE 22050 Hz, the original
`data/raw/LJSpeech-1.1/wavs/*.wav` files are already valid, unmodified
references at the exact rate the model and BigVGAN vocoder produce. No
separate reference-wav pipeline is needed.

Usage:
    python scripts/evaluate.py --ckpt logs/runs/.../checkpoints/last.ckpt \
        --preprocessed_dir data/preprocessed --filelist val.txt \
        --ljspeech_dir data/raw/LJSpeech-1.1 --out_dir logs/eval

VERSA is invoked as a subprocess (not imported) so `--help` parses even
without the `evaluate` extra installed, and so this project's own
torch/torchaudio pins never have to reconcile with VERSA's dependency tree at
import time. `configs/versa/cpu.yaml` (VERSA's own CPU-only example config:
mcd_f0, signal_metric, pesq, stoi) is used by default. No large pretrained
metric models get downloaded unless --versa_config points at a heavier one.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import torch
from tqdm import tqdm

from src.data.dataset import TTSDataset
from src.data.frontends import load_frontend_for_preprocessed_dir
from src.data.normalization import normalize_text
from src.lightning_module import TTSLightningModule


@torch.no_grad()
def _synthesize_all(args, model, frontend, preprocess_config) -> str:
    basenames, _, _, raw_texts = TTSDataset._load_filelist(os.path.join(args.preprocessed_dir, args.filelist))
    if args.limit:
        basenames, raw_texts = basenames[: args.limit], raw_texts[: args.limit]

    from src.vocoders import load_vocoder

    vocoder = load_vocoder(args.vocoder_checkpoint, device=args.device)

    pred_dir = os.path.join(args.out_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)

    import soundfile as sf

    for basename, raw_text in tqdm(list(zip(basenames, raw_texts)), desc="Synthesizing"):
        text = normalize_text(raw_text, preprocess_config.get("normalize", "none"))
        ids = torch.LongTensor([frontend.encode(text)])
        src_lens = torch.LongTensor([ids.shape[1]])
        output = model.model.synthesize(text_ids=ids, src_lens=src_lens, max_src_len=ids.shape[1])
        mel = output["mel"][0]
        wav = vocoder(mel.to(args.device).transpose(0, 1).unsqueeze(0))[0].cpu()
        sf.write(os.path.join(pred_dir, f"{basename}.wav"), wav.numpy(), vocoder.sampling_rate)

    return pred_dir


def _run_versa(pred_dir: str, gt_dir: str, versa_config: str, out_file: str):
    spec = importlib.util.find_spec("versa")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "versa is not installed. Run `uv sync --extra evaluate` first "
            "(installs from https://github.com/wavlab-speech/versa)."
        )
    versa_dir = list(spec.submodule_search_locations)[0]
    scorer = os.path.join(versa_dir, "bin", "scorer.py")

    cmd = [
        sys.executable, scorer,
        "--pred", pred_dir,
        "--gt", gt_dir,
        "--score_config", versa_config,
        "--output_file", out_file,
        "--io", "dir",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _summarize(out_file: str):
    if not os.path.isfile(out_file):
        print(f"VERSA finished but {out_file} was not found. Check its own console output above.")
        return

    totals: dict[str, list[float]] = {}
    with open(out_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    totals.setdefault(key, []).append(value)

    if not totals:
        print(f"Wrote {out_file} but could not parse per-utterance metrics from it. Inspect the file directly.")
        return

    print("\nMean metrics:")
    for key, values in sorted(totals.items()):
        print(f"  {key}: {sum(values) / len(values):.4f} (n={len(values)})")


def main(args):
    model = TTSLightningModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    model.to(args.device)

    frontend = load_frontend_for_preprocessed_dir(args.preprocessed_dir)
    with open(os.path.join(args.preprocessed_dir, "preprocess_config.json")) as f:
        preprocess_config = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = _synthesize_all(args, model, frontend, preprocess_config)

    gt_dir = os.path.join(args.ljspeech_dir, "wavs")
    out_file = os.path.join(args.out_dir, "versa_results.jsonl")
    _run_versa(pred_dir, gt_dir, args.versa_config, out_file)
    _summarize(out_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--preprocessed_dir", default="data/preprocessed")
    parser.add_argument("--filelist", default="val.txt")
    parser.add_argument("--ljspeech_dir", default="data/raw/LJSpeech-1.1", help="source of ground-truth wavs")
    parser.add_argument("--out_dir", default="logs/eval")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of utterances, for a quick smoke test")
    parser.add_argument("--vocoder_checkpoint", default=None, help="default: nvidia/bigvgan_v2_22khz_80band_256x")
    parser.add_argument("--versa_config", default="configs/versa/cpu.yaml")
    parser.add_argument("--device", default="cpu")
    return parser


def cli():
    """Entry point for the `tts-evaluate` uv tool (see [project.scripts]
    in pyproject.toml). Equivalent to `uv run python scripts/evaluate.py ...`."""
    main(build_parser().parse_args())


if __name__ == "__main__":
    cli()

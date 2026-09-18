"""Generate a tiny, synthetic LJSpeech-1.1-shaped dataset (a few sine-tone
.wav files + metadata.csv) so the pipeline can be smoke-tested end to end
without downloading the real ~2.6GB dataset. Used by .github/workflows/ci.yml
and available for local use the same way.

Usage:
    uv run python tests/make_fake_ljspeech.py [--out_dir data/raw/LJSpeech-1.1] [--n 12]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import soundfile as sf

SAMPLING_RATE = 22050

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a test.",
    "Preprocessing should work end to end.",
    "Continuous integration keeps the template honest.",
    "She sells seashells by the seashore.",
    "A generic front end for text to speech.",
]


def main(out_dir: str, n: int, seed: int):
    os.makedirs(os.path.join(out_dir, "wavs"), exist_ok=True)
    rng = np.random.default_rng(seed)

    lines = []
    for i in range(n):
        basename = f"LJ001-{i:04d}"
        text = _SENTENCES[i % len(_SENTENCES)]
        duration = rng.uniform(0.8, 1.6)
        t = np.linspace(0, duration, int(SAMPLING_RATE * duration), endpoint=False)
        freq = rng.uniform(150, 300)  # varied tone per utterance, not that it matters for a smoke test
        wav = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        sf.write(os.path.join(out_dir, "wavs", f"{basename}.wav"), wav, SAMPLING_RATE)
        lines.append(f"{basename}|{text}|{text}")

    with open(os.path.join(out_dir, "metadata.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} synthetic utterances to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw/LJSpeech-1.1")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.out_dir, args.n, args.seed)

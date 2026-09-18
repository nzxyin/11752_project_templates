"""Inference: text -> tokens (via the same text frontend used at preprocessing
time) -> acoustic model -> mel-spectrogram -> (optionally) BigVGAN -> waveform.

Usage:
    # mel only:
    python scripts/synthesize.py --ckpt logs/runs/.../checkpoints/last.ckpt \
        --text "The quick brown fox jumps over the lazy dog." --out mel.npy --plot

    # mel + audio (needs the `vocoder` extra: uv sync --extra vocoder):
    python scripts/synthesize.py --ckpt logs/runs/.../checkpoints/last.ckpt \
        --text "The quick brown fox jumps over the lazy dog." --out out.wav

The text frontend and normalization backend are auto-detected from
`preprocess_config.json`/`symbols.json` next to the checkpoint's training data
(or override with --preprocessed_dir) -- these must match training or token
ids will be nonsensical. With no vocoder resolvable, only the mel-spectrogram
is saved (as .npy, plus a .png if --plot).

--model_kwarg key=value (repeatable, JSON-decoded value) forwards arbitrary
model-specific inference args into BaseTTSModel.synthesize(**kwargs) -- e.g. a
Matcha-TTS ODE step count or an F5-TTS cfg_scale. If the loaded model sets
requires_reference_audio=True (F5-TTS-style voice cloning), --ref_audio and
--ref_text are required and forwarded as ref_mel/ref_text_ids.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.data.frontends import load_frontend_for_preprocessed_dir
from src.data.normalization import normalize_text
from src.lightning_module import TTSLightningModule
from src.utils.tools import plot_mel


def _resolve_preprocessed_dir(ckpt_path: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    # best-effort: look for preprocess_config.json relative to the checkpoint,
    # falling back to the common data/preprocessed/ location relative to cwd.
    for candidate in (
        os.path.join(os.path.dirname(ckpt_path), "..", "..", ""),
        os.path.join("data", "preprocessed"),
    ):
        if os.path.isfile(os.path.join(candidate, "preprocess_config.json")):
            return candidate
    return None


def _parse_model_kwargs(pairs: list[str] | None) -> dict:
    kwargs = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value  # plain string, e.g. --model_kwarg mode=fast
    return kwargs


@torch.no_grad()
def main(args):
    model = TTSLightningModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()

    preprocessed_dir = _resolve_preprocessed_dir(args.ckpt, args.preprocessed_dir)
    if preprocessed_dir is None:
        raise SystemExit(
            "Could not find preprocess_config.json/symbols.json to determine the text "
            "frontend this checkpoint was trained with -- pass --preprocessed_dir explicitly."
        )
    frontend = load_frontend_for_preprocessed_dir(preprocessed_dir)
    with open(os.path.join(preprocessed_dir, "preprocess_config.json")) as f:
        preprocess_config = json.load(f)

    text = normalize_text(args.text, preprocess_config.get("normalize", "none"))
    ids = torch.LongTensor([frontend.encode(text)])
    src_lens = torch.LongTensor([ids.shape[1]])

    kwargs = _parse_model_kwargs(args.model_kwarg)
    if model.model.requires_reference_audio:
        if not (args.ref_audio and args.ref_text):
            raise SystemExit("this model requires --ref_audio and --ref_text for synthesis")
        import librosa

        from src.vocoders.mel import bigvgan_mel_spectrogram

        ref_wav, _ = librosa.load(args.ref_audio, sr=preprocess_config["sampling_rate"])
        ref_mel = bigvgan_mel_spectrogram(
            torch.from_numpy(ref_wav).float().unsqueeze(0),
            preprocess_config["n_fft"], preprocess_config["n_mel_channels"], preprocess_config["sampling_rate"],
            preprocess_config["hop_length"], preprocess_config["win_length"], preprocess_config["fmin"], preprocess_config["fmax"],
        ).squeeze(0).transpose(0, 1)  # (T_mel, n_mel)
        kwargs["ref_mel"] = ref_mel.unsqueeze(0)
        ref_text = normalize_text(args.ref_text, preprocess_config.get("normalize", "none"))
        kwargs["ref_text_ids"] = torch.LongTensor([frontend.encode(ref_text)])

    output = model.model.synthesize(text_ids=ids, src_lens=src_lens, max_src_len=ids.shape[1], **kwargs)
    mel = output["mel"][0]  # (T_mel, n_mel)

    is_wav_out = args.out.lower().endswith(".wav")

    if is_wav_out:
        from src.vocoders import load_vocoder

        vocoder = load_vocoder(args.vocoder_checkpoint, device=args.device)
        wav = vocoder(mel.to(args.device).transpose(0, 1).unsqueeze(0))[0].cpu()  # (T_wav,)

        import soundfile as sf

        sf.write(args.out, wav.numpy(), vocoder.sampling_rate)
        print(f"Saved {wav.shape[0] / vocoder.sampling_rate:.2f}s of audio to {args.out}")
    else:
        mel_np = mel.cpu().numpy()
        np.save(args.out, mel_np)
        print(f"Saved mel-spectrogram {mel_np.shape} to {args.out} (no vocoder run -- .wav --out to vocode)")

    if args.plot:
        fig = plot_mel([mel.cpu().numpy().T], ["Synthesized Mel"])
        plot_path = os.path.splitext(args.out)[0] + "_mel.png"
        fig.savefig(plot_path)
        print(f"Saved plot to {plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", default="synthesized.wav", help=".wav for audio, .npy to save the mel array instead")
    parser.add_argument("--preprocessed_dir", default=None, help="default: auto-detected next to the checkpoint's training data")
    parser.add_argument("--vocoder_checkpoint", default=None, help="HF Hub repo id or local BigVGAN directory (default: nvidia/bigvgan_v2_22khz_80band_256x)")
    parser.add_argument("--model_kwarg", action="append", metavar="key=value", help="repeatable -- forwarded into BaseTTSModel.synthesize(**kwargs)")
    parser.add_argument("--ref_audio", default=None, help="reference audio path -- required if the model sets requires_reference_audio=True")
    parser.add_argument("--ref_text", default=None, help="reference audio's transcript -- required alongside --ref_audio")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot", action="store_true")
    return parser


def cli():
    """Entry point for the `tts-synthesize` uv tool (see [project.scripts]
    in pyproject.toml) -- equivalent to `uv run python scripts/synthesize.py ...`."""
    main(build_parser().parse_args())


if __name__ == "__main__":
    cli()

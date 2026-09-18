# Matcha-TTS Template (Hydra + PyTorch Lightning)

A [generic TTS front-end](../../tree/master) (see that branch's README for the
full picture), tailored to implementing **Matcha-TTS**: Mehta, Tu, Beskow,
Székely & Henter, ["Matcha-TTS: A fast TTS architecture with conditional flow
matching"](https://arxiv.org/abs/2309.03199) (ICASSP 2024). Built on
[Hydra](https://hydra.cc/) + [PyTorch Lightning](https://lightning.ai/), it
provides everything *around* the model: LJSpeech preprocessing at its
**native 22050 Hz**, a training/validation/testing loop, checkpointing +
TensorBoard/W&B logging, DDP/multi-node, text-to-mel-to-waveform inference
wired up to the official **BigVGANv2 22kHz** vocoder checkpoint, and
objective evaluation via the [VERSA](https://github.com/wavlab-speech/versa)
toolkit -- so you can focus on the model itself and get training, testing,
inference, and evaluation for free.

**This template does not implement Matcha-TTS.** `src/models/matcha.py`
(`MatchaTTSPlaceholder`) is a minimal, deliberately-naive stand-in (no
Monotonic Alignment Search, no duration predictor, no flow-matching decoder)
that exists only so every command below actually runs out of the box, as an
integration smoke test -- see [Implementing Matcha-TTS](#implementing-matcha-tts)
below for the shape of what's missing.

## Project layout

```
configs/                 Hydra configs (composable via CLI overrides)
  config.yaml             top-level: composes the groups below
  paths/default.yaml       all filesystem paths, referenced via ${paths.xxx}
  data/ljspeech.yaml        DataModule + batching options
  model/matcha.yaml         PLACEHOLDER model + optimizer/scheduler config (see "The model contract")
  trainer/{default,ddp}.yaml   pytorch_lightning.Trainer args (single-device / multi-GPU+node)
  callbacks/default.yaml    checkpointing (monitors val/loss), LR monitor, progress bar
  logger/{tensorboard,wandb,both}.yaml
  experiment/debug.yaml     tiny CPU smoke-test override
  versa/cpu.yaml            VERSA's CPU-only objective-metric config (scripts/evaluate.py default)

src/
  train.py                 Hydra entry point
  lightning_module.py       TTSLightningModule (generic LightningModule): wraps ANY
                             src.models.base.BaseTTSModel, handles logging,
                             AdamW(fused)+Hydra-instantiated scheduler, checkpoint hparams
  models/
    base.py                  BaseTTSModel: the contract a real model must satisfy
    matcha.py                  PLACEHOLDER model (embedding + tiny Transformer encoder +
                               naive length-matched linear mel projection) -- replace this
  data/
    frontends/                pluggable text -> token frontends (see below)
    normalization.py          optional text-normalization pre-pass (not needed for LJSpeech)
    dataset.py                 TTSDataset + collate_fn (text ids + mel, generic to any model)
    datamodule.py               TTSDataModule (+ length-bucketed batch sampler)
  vocoders/                  mel -> waveform via BigVGANv2 22kHz
    mel.py                     bit-exact mel extraction matching that checkpoint's convention
    bigvgan/                   vendored BigVGAN generator (NVIDIA/BigVGAN, MIT) -- see its
                                THIRD_PARTY_NOTICES.md
    bigvgan_vocoder.py          BigVGANVocoder: load a HF Hub checkpoint, run mel -> wav
  utils/                      mask/padding helpers, mel plotting, rank-zero logger

scripts/
  preprocess.py             LJSpeech (native 22050 Hz) -> mel + phoneme/token filelists
                             uv tool: tts-preprocess
  synthesize.py             checkpoint + text -> mel-spectrogram -> (optionally) waveform
                             uv tool: tts-synthesize
  evaluate.py                checkpoint + val split -> synthesized wavs -> VERSA objective metrics
                             uv tool: tts-evaluate
```

`src/train.py` is the `tts-train` uv tool -- see [Setup](#setup-uv) for all four.

## The model contract

Everything above is designed around one interface,
`src.models.base.BaseTTSModel` (an `nn.Module`):

```python
class BaseTTSModel(nn.Module, abc.ABC):
    requires_reference_audio: bool = False  # True for F5-TTS-style voice cloning

    def forward(self, batch: dict) -> dict:
        """Teacher-forced train/val step. batch = collate_fn's output: texts,
        src_lens, max_src_len, mels, mel_lens, max_mel_len, speakers, ids, raw_texts.
        Must return {"loss": 0-d tensor, ...any other scalars to auto-log}.
        Optionally include "mel_pred" (B,T_mel,n_mel) to enable the default
        GT-vs-predicted mel image logged every validation epoch."""

    def synthesize(self, text_ids, src_lens, max_src_len=None, **kwargs) -> dict:
        """Inference, no ground-truth mel. kwargs are open-ended and
        model-specific. Must return {"mel": (B,T_mel,n_mel)}."""

    def log_artifacts(self, batch: dict, outputs: dict) -> dict:
        """Optional extra validation figures beyond the default mel pair."""
```

**To plug in a real model**: write an `nn.Module` subclassing `BaseTTSModel`
(e.g. edit `src/models/matcha.py` in place), and point
`configs/model/matcha.yaml`'s `network._target_` at it, e.g.:

```yaml
_target_: src.lightning_module.TTSLightningModule
network:
  _target_: src.models.matcha.MatchaTTS
  hidden: 192
  # ... your architecture's hyperparams
optimizer: { _target_: torch.optim.AdamW, lr: 1.0e-4, ... }
scheduler: { _target_: torch.optim.lr_scheduler.OneCycleLR, ... }
```

`src/lightning_module.py`, the dataset/datamodule, and all four CLI scripts
need **no changes** -- `TTSLightningModule` only ever calls `self.model(batch)`
and `self.model.synthesize(...)`, so any conforming implementation slots in by
switching `model=<name>` on the CLI (`uv run tts-train model=matcha`).

- **FastSpeech2**: duration predictor + length regulator, all internal to the
  subclass; `forward` returns `mel_pred` for the default validation image.
- **Matcha-TTS**: Monotonic-Alignment-Search duration + a flow-matching
  decoder -- fits cleanly since alignment/duration stays fully internal.
- **F5-TTS**: set `requires_reference_audio = True`; `scripts/synthesize.py`
  then requires `--ref_audio`/`--ref_text` and forwards `ref_mel`/`ref_text_ids`
  as `synthesize(**kwargs)`. A diffusion/flow model whose training step
  doesn't produce a cheap mel sample can simply omit `"mel_pred"` from
  `forward`'s return dict -- `tts-evaluate`'s full `synthesize()` call is the
  real quality check for those.

## Text frontends

Raw text becomes model input token ids via a pluggable frontend
(`src/data/frontends/`), selected once at preprocessing time and then
auto-detected at inference/evaluation time from `preprocess_config.json` +
`symbols.json` next to the checkpoint's training data:

| `--frontend` | Symbol set | System deps | Notes |
|---|---|---|---|
| `g2p_en` (default) | ARPAbet-39 (fixed) | none beyond the `preprocess`/`synthesize` extras | English only, via `g2p_en`/CMUdict |
| `phonemizer` | IPA (data-driven, persisted to `symbols.json`) | `espeak-ng` system binary | what Matcha-TTS-style implementations typically expect; multi-lingual |

Optionally normalize text before phonemization with `--normalize {none,nemo}`
(`src/data/normalization.py`) -- **not needed for LJSpeech**, whose
`metadata.csv` transcripts are already normalized; this is an extension point
for a messier, non-LJSpeech dataset later. The `nemo` backend needs
Linux/WSL/conda-forge (its `pynini` dependency has no native Windows wheels).

## Setup ([uv](https://docs.astral.sh/uv/))

```bash
uv sync --extra preprocess --extra synthesize --extra vocoder
```

This creates `.venv/` and installs everything from `pyproject.toml`, including
four uv-runnable console-script tools (`[project.scripts]`):

| Tool | Equivalent to |
|---|---|
| `uv run tts-preprocess ...` | `uv run python scripts/preprocess.py ...` |
| `uv run tts-train ...` | `uv run python -m src.train ...` |
| `uv run tts-synthesize ...` | `uv run python scripts/synthesize.py ...` |
| `uv run tts-evaluate ...` | `uv run python scripts/evaluate.py ...` |

Every command below uses the tool form; the `python -m` / `python scripts/...`
form on the right works identically if you prefer it (or want an IDE debugger
attached to a specific file).

Optional extras, add as needed:
- `--extra preprocess` / `--extra synthesize`: `g2p_en`, the default text frontend.
- `--extra phonemizer`: the alternative IPA frontend (`--frontend phonemizer`).
  Also needs the **espeak-ng system binary** (not pip-installable):
  - Debian/Ubuntu: `sudo apt install espeak-ng`
  - macOS: `brew install espeak-ng`
  - Windows: download the installer from the
    [espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases)
    and add it to `PATH`; if `phonemizer` still can't find it, set
    `PHONEMIZER_ESPEAK_LIBRARY` to the path of `libespeak-ng.dll`.
- `--extra normalize`: `nemo_text_processing` (`--normalize nemo`). **Not
  needed for LJSpeech.** Requires Linux, WSL, or conda-forge -- `pynini` has
  no native Windows wheels, so this extra will fail to install on native Windows.
- `--extra vocoder`: actually loading the BigVGANv2 pretrained checkpoint
  (`huggingface_hub`) to turn a predicted mel into audio. Not needed for
  training or preprocessing (`src/vocoders/mel.py`'s mel extraction only needs
  torch/torchaudio/librosa, already core deps).
- `--extra evaluate`: [VERSA](https://github.com/wavlab-speech/versa)
  (`tts-evaluate`), installed from git. Invoked as a subprocess, so a heavier
  VERSA dependency tree never has to reconcile with this project's own pins.
- `--extra wandb`: W&B logging.

`torch`/`torchaudio` install CPU wheels by default via `uv sync`; for a CUDA
build, follow [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
(e.g. add the appropriate `[[tool.uv.index]]` for your CUDA version) before syncing.

Each git branch's `pyproject.toml`/`uv.lock` is independent -- if you implement
FastSpeech2/Matcha-TTS/F5-TTS on its own branch with extra dependencies
(`uv add numba scipy`, say), `uv sync` on that branch installs exactly what
that model needs without touching this one.

> If you do call the underlying script directly instead of the tool, invoke
> training as a module (`python -m src.train`, not `python src/train.py`) so
> `src` resolves as a package from the repo root. `tts-train` handles this
> itself regardless of how it's invoked (see `_CONFIG_DIR` in `src/train.py`).

## Data / Preprocessing

Download [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) and extract it to
`data/raw/LJSpeech-1.1/` (contains `metadata.csv` and `wavs/`), then:

```bash
uv run tts-preprocess --ljspeech_dir data/raw/LJSpeech-1.1 --out_dir data/preprocessed
```

LJSpeech is kept at its **native 22050 Hz** -- no resampling. Mel-spectrograms
are extracted matching the exact convention of BigVGANv2's official 22kHz
checkpoint (`nvidia/bigvgan_v2_22khz_80band_256x`: 80 mels, n_fft=1024,
hop=256, win=1024) so a model trained to reconstruct these mels sounds right
once vocoded.

This writes `mel/`, `train.txt`, `val.txt`, `speakers.json`, `symbols.json`,
`stats.json`, and `preprocess_config.json` (records the `--frontend`/
`--language`/`--normalize` used, so `scripts/synthesize.py`/`scripts/evaluate.py`
auto-detect a matching text frontend) under `data/preprocessed/`.

Use the IPA frontend instead: `uv run tts-preprocess --frontend phonemizer --language en-us ...`
(needs `--extra phonemizer` + the espeak-ng system binary, see [Setup](#setup-uv)).

## Training

```bash
uv run tts-train
```

Everything is overridable from the CLI (Hydra):

```bash
uv run tts-train trainer.max_steps=50000 data.batch_size=32
uv run tts-train model.network.hidden=384                  # placeholder model's hyperparams
uv run tts-train logger=wandb                     # switch TensorBoard -> W&B
uv run tts-train logger=both                      # both simultaneously
uv run tts-train experiment=debug                  # tiny CPU run to check shapes
uv run tts-train ckpt_path=logs/runs/.../last.ckpt # resume
```

Logs and checkpoints land under `logs/runs/<name>/<timestamp>/` (see
`configs/paths/default.yaml`, `configs/callbacks/default.yaml`, which monitors
`val/loss` -- the contract's mandatory loss key). View TensorBoard with:

```bash
uv run tensorboard --logdir logs/tensorboard
```

### Optimizer / schedule

Optimizer and scheduler are ordinary Hydra-instantiated objects
(`configs/model/matcha.yaml`'s `optimizer`/`scheduler` blocks) -- any
`torch.optim` optimizer and `torch.optim.lr_scheduler` scheduler works via
`_target_`; nothing in `TTSLightningModule` assumes a particular choice.
Different TTS papers favor different setups (Adam with a Transformer-style
warmup + inverse-square-root decay, as in the original FastSpeech2/
Transformer-TTS line of work; plain AdamW with a fixed or linear-warmup
schedule, common in more recent flow-matching models) -- pick whatever the
paper you're implementing actually uses, or whatever works for you.

The placeholder ships AdamW with the fused CUDA kernel
(`model.optimizer.fused=true`) + a
[OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
1cycle policy (`model.scheduler`), whose `total_steps` is tied to
`trainer.max_steps` so one CLI override keeps both in sync -- purely as one
concrete, runnable default, not a recommendation for any particular
architecture. `TTSLightningModule.configure_optimizers` automatically falls
back to `fused=false` when training on CPU (e.g. `experiment=debug`), so the
same config works everywhere without editing it.

> **Resuming**: OneCycleLR's checkpointed state (including `total_steps`)
> overwrites whatever the new run's config says, so resuming with a *different*
> `trainer.max_steps` than the original run will raise (`Tried to step N times...`).
> Resuming with the *same* `max_steps` (the normal crash-recovery case) works
> fine. If you deliberately want to extend a finished run, either start a fresh
> run with the larger budget, or swap in a different, resumable-by-design
> scheduler (e.g. cosine-with-restarts).

### Multi-GPU / multi-node (DDP)

```bash
uv run tts-train trainer=ddp                                             # all GPUs on this machine
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$RANK --master_addr=$ADDR \
    -m src.train trainer=ddp trainer.num_nodes=2                          # 2 nodes x 8 GPUs (torchrun needs a module, not the tool)
```

See `configs/trainer/ddp.yaml` for details. Effective global batch size =
`data.batch_size * devices * trainer.num_nodes * trainer.accumulate_grad_batches`
-- scale `data.batch_size` down or `model.optimizer.lr` up accordingly. Under
SLURM, Lightning's `SLURMEnvironment` autodetects node rank/addr/port and
`trainer=ddp` works unchanged inside an `srun`/`sbatch` script instead of `torchrun`.

## Vocoder

An acoustic model built on this template predicts mel-spectrograms, not
audio -- turning those into a waveform is a separate vocoder model. This
template wires up the official pretrained
**[BigVGANv2](https://github.com/NVIDIA/BigVGAN)** (NVIDIA, MIT license) 22kHz
checkpoint (`nvidia/bigvgan_v2_22khz_80band_256x`): its generator architecture
is vendored into `src/vocoders/bigvgan/` (see that directory's
`THIRD_PARTY_NOTICES.md` for exactly what was copied/changed and why) so
`BigVGANVocoder` can load the official checkpoint straight from the Hugging
Face Hub without depending on the full NVIDIA/BigVGAN repo.

**Why the mel convention has to match exactly**: BigVGAN was trained on
mel-spectrograms extracted a specific way (STFT framing, mel filterbank
normalization, log-compression clamp) -- see `src/vocoders/mel.py`, verified
bit-exact against BigVGAN's own reference implementation.
`scripts/preprocess.py` always extracts training targets this same way, so a
model trained on this template's preprocessed mels sounds right once vocoded,
with no flags to keep in sync.

## Inference

```bash
# mel + audio (needs the `vocoder` extra: uv sync --extra vocoder):
uv run tts-synthesize --ckpt logs/runs/.../checkpoints/last.ckpt \
    --text "The quick brown fox jumps over the lazy dog." --out out.wav --plot

# mel only (no vocoder download needed):
uv run tts-synthesize --ckpt logs/runs/.../checkpoints/last.ckpt \
    --text "The quick brown fox jumps over the lazy dog." --out mel.npy --plot
```

The text frontend and normalization backend are auto-detected from
`preprocess_config.json`/`symbols.json` next to the checkpoint's training data
(override with `--preprocessed_dir`). `--vocoder_checkpoint` picks a different
Hugging Face Hub repo id (or a local directory) than BigVGANv2's default
22kHz checkpoint.

`--model_kwarg key=value` (repeatable, JSON-decoded value) forwards arbitrary
model-specific inference args into `BaseTTSModel.synthesize(**kwargs)` -- e.g.
a Matcha-TTS ODE step count or an F5-TTS `cfg_scale`. If the loaded model sets
`requires_reference_audio=True` (F5-TTS-style voice cloning), pass
`--ref_audio <wav>` and `--ref_text "..."`.

## Evaluation

Objective metrics via the [VERSA](https://github.com/wavlab-speech/versa)
toolkit (needs `--extra evaluate` + `--extra vocoder`):

```bash
uv run tts-evaluate --ckpt logs/runs/.../checkpoints/last.ckpt \
    --preprocessed_dir data/preprocessed --filelist val.txt \
    --ljspeech_dir data/raw/LJSpeech-1.1 --out_dir logs/eval
```

Synthesizes every utterance in `--filelist`, vocodes it, and scores the result
against LJSpeech's own ground-truth wavs (valid references as-is, since
preprocessing never resamples away from LJSpeech's native rate). Defaults to
`configs/versa/cpu.yaml`, VERSA's own lightweight CPU-only metric set (mel
cepstral distortion, signal metrics, PESQ, STOI -- no large pretrained metric
models downloaded). Pass `--versa_config` to point at a heavier VERSA config
(e.g. one that adds UTMOS/DNSMOS/speaker-similarity) for a slower but more
thorough evaluation. `--limit N` caps the number of utterances for a quick
smoke test.

## Implementing Matcha-TTS

`src/models/matcha.py`'s `MatchaTTSPlaceholder` is where the real model goes.
From the paper, Matcha-TTS is:

```
phonemes -> [text encoder]
         -> [duration predictor, trained against Monotonic Alignment Search
             (MAS) durations computed between encoder output and target mel]
         -> [length regulator, using predicted durations at inference]
         -> [U-Net decoder trained with (optimal-transport) conditional flow
             matching -- predicts a vector field, not the mel directly]
         -> ODE integration (e.g. a handful of Euler steps) from noise
            -> mel-spectrogram
```

A few things this means for fitting it into `BaseTTSModel`
(`forward(batch) -> dict`, `synthesize(text_ids, src_lens, **kwargs) -> dict`):

- **Alignment is self-contained.** Unlike FastSpeech2, Matcha-TTS doesn't need
  an external forced aligner or precomputed durations -- MAS runs at train
  time directly between the encoder output and the ground-truth mel (similar
  to Glow-TTS/VITS), so `forward()` can compute everything from the batch's
  `texts`/`mels` as-is; no preprocessing changes needed for this part.
- **The decoder isn't predicting a mel directly.** Flow matching trains the
  network to predict a vector field along a probability path from noise to
  data; `forward()`'s `"loss"` is the flow-matching objective (e.g. an
  OT-CFM regression loss), not a direct mel reconstruction loss like
  FastSpeech2's. Because of this, a cheap `"mel_pred"` isn't a free byproduct
  of a training step the way it is for a regression model -- it's fine to
  omit it and rely on `tts-evaluate`'s full `synthesize()` call (which does
  run the ODE solver) as the real quality check instead.
- **`synthesize()`'s `**kwargs`** is a natural place for inference-time knobs
  the paper exposes: number of ODE (sampling) steps, temperature, guidance
  scale, etc. -- forwarded from `scripts/synthesize.py --model_kwarg key=value`.
- The vendored `src/vocoders/bigvgan_vocoder.py` and `src/vocoders/mel.py`
  need no changes -- Matcha-TTS's job is only to predict a mel matching that
  convention (`configs/data/ljspeech.yaml`'s `n_mel_channels: 80`).

## Notes / where to extend

- **A different model**: see [The model contract](#the-model-contract) above --
  that's the whole point of this template.
- **Multi-speaker**: set `data.multi_speaker=true`; `MatchaTTSPlaceholder` already
  demonstrates the pattern (a speaker embedding added to the encoder input) --
  a real implementation should do the same, plus a per-speaker `speakers.json`
  during preprocessing (currently LJSpeech-only, single speaker).
- **A different text frontend / language**: add a new
  `src/data/frontends/<name>_frontend.py` subclassing `TextFrontend`, register
  it in `src/data/frontends/__init__.py`'s `FRONTENDS` dict.
- **A different sample rate / vocoder**: update `SAMPLING_RATE`/`N_FFT`/
  `HOP_LENGTH`/`WIN_LENGTH`/`N_MEL_CHANNELS`/`MEL_FMAX` in
  `scripts/preprocess.py` (and `data.n_mel_channels` in
  `configs/data/ljspeech.yaml`) to match, update `src/vocoders/mel.py` and
  `src/vocoders/__init__.py`'s default checkpoint, and re-preprocess.

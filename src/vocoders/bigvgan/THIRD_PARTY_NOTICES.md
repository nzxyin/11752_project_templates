# Third-party code

The files in this directory (`src/vocoders/bigvgan/`) are vendored from
[NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN), MIT licensed, fetched
2026-08-19. They implement the BigVGAN generator architecture so its official
pretrained checkpoints (this template defaults to `nvidia/bigvgan_v2_22khz_80band_256x`
on the Hugging Face Hub) can be loaded and run for inference
(`BigVGAN.from_pretrained(...)`) without depending on the full `NVIDIA/BigVGAN`
repo/package.

BigVGAN itself adapts code from [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
(MIT) and [junjun3518/alias-free-torch](https://github.com/junjun3518/alias-free-torch)
(Apache-2.0); see the header comment in each file for its specific upstream origin.

**Changes made here, relative to upstream:**
- `bigvgan.py`, `alias_free_activation/torch/{__init__,act,resample}.py`: absolute
  sibling imports (`import activations`, `from alias_free_activation.torch...`)
  rewritten as relative imports (`from . import activations`, `from .resample
  import ...`) so this can live as a normal Python subpackage instead of
  requiring its directory to be manually added to `sys.path`.
- `bigvgan.py`: the optional fused CUDA kernel path
  (`alias_free_activation/cuda/`, which JIT-compiles via `nvcc`+`ninja`) is not
  vendored. `use_cuda_kernel=True` now raises `NotImplementedError` with a
  pointer to the upstream repo instead of silently failing an import.
  `use_cuda_kernel=False` (the default) is functionally identical to upstream
  and is all `src/vocoders/bigvgan_vocoder.py` uses.
- `utils.py`: trimmed to just `init_weights`/`get_padding` (the two functions
  the generator needs). Upstream's training-only checkpoint-scanning and
  matplotlib-plotting helpers are dropped to avoid pulling in unneeded deps.
- `meldataset.py`'s `mel_spectrogram` function is *not* copied verbatim here.
  It is reproduced (same math, adjustable defaults) as `bigvgan_mel_spectrogram`
  in `src/vocoders/mel.py`.

No other logic was changed. See the BigVGAN repository for its full license
text and `incl_licenses/` third-party notices.

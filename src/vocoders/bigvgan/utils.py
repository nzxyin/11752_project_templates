# Trimmed from NVIDIA/BigVGAN's utils.py (https://github.com/NVIDIA/BigVGAN,
# MIT license), commit as of 2026-08-19 -- see ../THIRD_PARTY_NOTICES.md.
# Only `init_weights`/`get_padding` are kept (the two the generator needs for
# inference); the training-only checkpoint-scanning and matplotlib plotting
# helpers from upstream are dropped to avoid pulling in unneeded dependencies.
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

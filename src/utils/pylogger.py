"""Rank-zero-aware python logger, so multi-GPU training doesn't spam duplicate log lines."""
import logging

from pytorch_lightning.utilities import rank_zero_only


def get_pylogger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger

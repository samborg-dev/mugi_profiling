import torch

EXP_BIAS = 128
N_BINS = 256

WINDOW_LO_EXP = -15
WINDOW_HI_EXP = 14
WINDOW_BINS = WINDOW_HI_EXP - WINDOW_LO_EXP + 1

LEGACY_N_BINS = 32
LEGACY_EXP_BIAS = 16


def analysis_window(hist: torch.Tensor) -> torch.Tensor:
    if hist.dim() != 1:
        raise ValueError(f"exponent histogram must be 1-D, got shape {tuple(hist.shape)}")

    if len(hist) == N_BINS:
        bias = EXP_BIAS
    elif len(hist) == LEGACY_N_BINS:
        bias = LEGACY_EXP_BIAS
    else:
        raise ValueError(
            f"unrecognised exponent histogram length {len(hist)}; expected "
            f"{N_BINS} (current) or {LEGACY_N_BINS} (legacy)"
        )

    lo = bias + WINDOW_LO_EXP
    window = hist[lo:lo + WINDOW_BINS]
    if len(window) != WINDOW_BINS:
        raise ValueError(
            f"histogram of length {len(hist)} is too short for the analysis "
            f"window [{WINDOW_LO_EXP}, {WINDOW_HI_EXP}]"
        )
    return window


def normalize_window(window: torch.Tensor, path: str = "") -> torch.Tensor:
    total = window.sum()
    if total == 0:
        where = f" ({path})" if path else ""
        raise ValueError(
            f"exponent histogram{where} is empty across the analysis window "
            f"[{WINDOW_LO_EXP}, {WINDOW_HI_EXP}]; nothing to select a LUT window from"
        )
    return (window / total) * 100


def index_to_exp(index: int) -> int:
    return index + WINDOW_LO_EXP


def exp_to_index(exp: int) -> int:
    return exp - WINDOW_LO_EXP

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from exponent_bins import analysis_window, exp_to_index, index_to_exp, normalize_window
from profiling_api.window import AutoCluster, TopKWindow, WindowCriterion, WindowStrategy
from profiling_api.windows import (ATTENTION, DEFAULT_GROUP_SIZE, FFN, FfnWindow,
                                   SoftmaxWindow, WindowAssignment)

SEED_SOURCE_HISTOGRAM = "histogram"
SEED_SOURCE_FALLBACK = "fallback"

FALLBACK_ANCHOR = 2
FALLBACK_ANCHOR_SIDE = "max"

SEED_ANCHOR_BOUNDS = (-7, 7)

_LAYER_DIR = re.compile(r"^layer_(\d+)$")
_SEQ_LEN_FILE = re.compile(r"^seq_len_(\d+)\.pt$")


@dataclass(frozen=True)
class LayerSeed:
    layer: int
    site: str
    window: object
    source: str
    max_exp: Optional[int] = None
    min_exp: Optional[int] = None
    cluster: Optional[str] = None
    detail: str = ""

    @property
    def from_histogram(self) -> bool:
        return self.source == SEED_SOURCE_HISTOGRAM

    def to_row(self) -> dict:
        return {
            "layer": self.layer,
            "site": self.site,
            "source": self.source,
            "anchor": getattr(self.window, "anchor", None),
            "anchor_side": getattr(self.window, "anchor_side", None),
            "exp_dim": getattr(self.window, "exp_dim", None),
            "group_size": getattr(self.window, "group_size", None),
            "profiled_max_exp": self.max_exp,
            "profiled_min_exp": self.min_exp,
            "cluster": self.cluster,
            "detail": self.detail,
        }


def _split(path: str) -> List[str]:
    return [p for p in re.split(r"[\\/]", path) if p]


def find_exp_dist_root(profile_root: str, nonlinear: str = "softmax",
                       stage: str = "pre_softmax") -> Optional[str]:
    if not profile_root or not os.path.isdir(profile_root):
        return None

    matches = []
    for dirpath, dirnames, _ in os.walk(profile_root):
        if os.path.basename(dirpath) != "exp_dist":
            continue
        parts = _split(dirpath)
        if stage and stage not in parts:
            continue
        if nonlinear and nonlinear not in parts:
            continue
        matches.append(dirpath)

    return sorted(matches)[0] if matches else None


def _seq_len_files(layer_dir: str) -> Dict[int, List[str]]:
    by_seq_len: Dict[int, List[str]] = {}
    for dirpath, _, filenames in os.walk(layer_dir):
        for name in filenames:
            match = _SEQ_LEN_FILE.match(name)
            if match:
                by_seq_len.setdefault(int(match.group(1)), []).append(
                    os.path.join(dirpath, name))
    return by_seq_len


class HistogramSeeder:
    def __init__(self, profile_root: str, nonlinear: str = "softmax",
                 stage: str = "pre_softmax", exp_dim: int = 16,
                 group_size: int = DEFAULT_GROUP_SIZE,
                 strategy: WindowStrategy = None, criterion: WindowCriterion = None,
                 fallback: SoftmaxWindow = None,
                 anchor_bounds: Tuple[int, int] = SEED_ANCHOR_BOUNDS):
        self.profile_root = profile_root
        self.nonlinear = nonlinear
        self.stage = stage
        self.exp_dim = exp_dim
        self.group_size = group_size
        self.strategy = strategy or TopKWindow()
        self.criterion = criterion or AutoCluster()
        self.anchor_bounds = anchor_bounds
        self.fallback = fallback or SoftmaxWindow(
            exp_dim=exp_dim, anchor=FALLBACK_ANCHOR,
            anchor_side=FALLBACK_ANCHOR_SIDE, group_size=group_size)
        self.exp_dist_root = find_exp_dist_root(profile_root, nonlinear, stage)

    def layer_dir(self, layer: int) -> Optional[str]:
        if self.exp_dist_root is None:
            return None
        return os.path.join(self.exp_dist_root, f"layer_{int(layer)}")

    def available_layers(self) -> List[int]:
        if self.exp_dist_root is None:
            return []
        layers = []
        for name in os.listdir(self.exp_dist_root):
            match = _LAYER_DIR.match(name)
            if match and os.path.isdir(os.path.join(self.exp_dist_root, name)):
                layers.append(int(match.group(1)))
        return sorted(layers)

    def seed_layer(self, layer: int, site: str = ATTENTION) -> LayerSeed:
        if self.exp_dist_root is None:
            return self._fallback(
                layer, site,
                f"no exp_dist directory for {self.nonlinear}/{self.stage} under "
                f"{self.profile_root!r}")

        layer_dir = self.layer_dir(layer)
        if not os.path.isdir(layer_dir):
            return self._fallback(layer, site, f"no histogram directory {layer_dir}")

        by_seq_len = _seq_len_files(layer_dir)
        if not by_seq_len:
            return self._fallback(layer, site, f"no seq_len_*.pt under {layer_dir}")

        seq_len = max(by_seq_len)
        paths = sorted(by_seq_len[seq_len])

        try:
            hist = self._load(paths)
            window = normalize_window(analysis_window(hist), paths[0])
            min_idx, max_idx = self.strategy.place(window, self.nonlinear)
            cluster = self.criterion.cluster(*self._cluster_sums(window))
        except (ValueError, RuntimeError, IndexError, TypeError) as e:
            return self._fallback(layer, site, f"{type(e).__name__}: {e}")

        max_exp = index_to_exp(int(max_idx))
        min_exp = index_to_exp(int(min_idx))
        placed = SoftmaxWindow.from_profile(max_exp=max_exp, min_exp=min_exp,
                                            cluster=cluster, exp_dim=self.exp_dim,
                                            group_size=self.group_size)
        clamped, detail = self._clamp(placed, seq_len, len(paths))

        return LayerSeed(layer=int(layer), site=site, window=clamped,
                         source=SEED_SOURCE_HISTOGRAM, max_exp=max_exp,
                         min_exp=min_exp, cluster=cluster, detail=detail)

    def seed_assignment(self, n_layers: int, ffn: FfnWindow = None
                        ) -> Tuple[WindowAssignment, List[LayerSeed]]:
        ffn = ffn or FfnWindow(exp_dim=self.exp_dim, pos_anchor=FALLBACK_ANCHOR,
                               neg_anchor=FALLBACK_ANCHOR, group_size=self.group_size)
        assignment = WindowAssignment()
        seeds = []
        for layer in range(int(n_layers)):
            seed = self.seed_layer(layer, ATTENTION)
            assignment.set(layer, ATTENTION, seed.window)
            assignment.set(layer, FFN, ffn)
            seeds.append(seed)
        return assignment, seeds

    def _load(self, paths: Iterable[str]) -> torch.Tensor:
        total = None
        for path in paths:
            hist = torch.load(path, map_location="cpu")
            if not torch.is_tensor(hist):
                raise TypeError(f"{path} holds a {type(hist).__name__}, not a tensor")
            hist = hist.detach().to(torch.float64).cpu()
            total = hist if total is None else total + hist
        return total

    def _cluster_sums(self, window: torch.Tensor) -> Tuple[float, float]:
        width = min(self.exp_dim, len(window))
        sums = torch.tensor([window[i:i + width].sum()
                             for i in range(len(window) - width + 1)])
        best = int(sums.argmax().item())

        max_i = exp_to_index(index_to_exp(best + width - 1))
        min_i = exp_to_index(index_to_exp(best))
        max_sum = window[max(0, max_i - width):max_i].sum().item()
        min_sum = window[min_i:min_i + width].sum().item()
        return max_sum, min_sum

    def _clamp(self, window: SoftmaxWindow, seq_len: int,
               n_files: int) -> Tuple[SoftmaxWindow, str]:
        detail = f"seq_len={seq_len} files={n_files}"
        lo, hi = self.anchor_bounds
        if lo <= window.anchor <= hi:
            return window, detail

        anchor = min(hi, max(lo, window.anchor))
        return (SoftmaxWindow(exp_dim=window.exp_dim, anchor=anchor,
                              anchor_side=window.anchor_side,
                              group_size=window.group_size),
                f"{detail} anchor {window.anchor} clamped to {anchor} "
                f"(searchable range [{lo}, {hi}])")

    def _fallback(self, layer: int, site: str, detail: str) -> LayerSeed:
        return LayerSeed(layer=int(layer), site=site, window=self.fallback,
                         source=SEED_SOURCE_FALLBACK, detail=detail)


def fallback_assignment(n_layers: int, exp_dim: int = 16,
                        group_size: int = DEFAULT_GROUP_SIZE,
                        attention: SoftmaxWindow = None, ffn: FfnWindow = None
                        ) -> Tuple[WindowAssignment, List[LayerSeed]]:
    attention = attention or SoftmaxWindow(exp_dim=exp_dim, anchor=FALLBACK_ANCHOR,
                                           anchor_side=FALLBACK_ANCHOR_SIDE,
                                           group_size=group_size)
    ffn = ffn or FfnWindow(exp_dim=exp_dim, pos_anchor=FALLBACK_ANCHOR,
                           neg_anchor=FALLBACK_ANCHOR, group_size=group_size)

    assignment = WindowAssignment()
    seeds = []
    for layer in range(int(n_layers)):
        assignment.set(layer, ATTENTION, attention)
        assignment.set(layer, FFN, ffn)
        seeds.append(LayerSeed(layer=layer, site=ATTENTION, window=attention,
                               source=SEED_SOURCE_FALLBACK,
                               detail="seeding disabled"))
    return assignment, seeds


def seeds_to_rows(seeds: Iterable[LayerSeed]) -> List[dict]:
    return [seed.to_row() for seed in seeds]

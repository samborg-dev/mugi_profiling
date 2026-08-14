import os
import re
from dataclasses import dataclass, asdict

import torch
import yaml

from profile_distribution import topk_window_until_threshold, profile_tensor, get_seq_len
from exponent_bins import analysis_window, exp_to_index, index_to_exp, normalize_window


@dataclass
class WindowSpec:
    argmax_value: int
    tensor: list
    max_exp: int
    min_exp: int
    max_exp_unclamped: int
    min_exp_unclamped: int
    mean: float
    median: float
    centroid: float
    cluster: str


class WindowStrategy:
    def place(self, tensor: torch.Tensor, nonlinear: str):
        raise NotImplementedError


class TopKWindow(WindowStrategy):
    def place(self, tensor: torch.Tensor, nonlinear: str):
        threshold = 0.92 if nonlinear == "softmax" else 0.5
        topk_indices, _ = topk_window_until_threshold(tensor, threshold=threshold)
        while topk_indices.shape[0] > 5:
            threshold -= 0.01
            topk_indices, _ = topk_window_until_threshold(tensor, threshold=threshold)
        while topk_indices.shape[0] < 4:
            threshold += 0.01
            topk_indices, _ = topk_window_until_threshold(tensor, threshold=threshold)
        return topk_indices.min().item(), topk_indices.max().item()


class WindowCriterion:
    def cluster(self, max_sum: float, min_sum: float) -> str:
        raise NotImplementedError


class MaxCluster(WindowCriterion):
    def cluster(self, max_sum, min_sum):
        return "max_cluster"


class MinCluster(WindowCriterion):
    def cluster(self, max_sum, min_sum):
        return "min_cluster"


class AutoCluster(WindowCriterion):
    def cluster(self, max_sum, min_sum):
        return "min_cluster" if min_sum > max_sum else "max_cluster"


_STRATEGIES = {"topk": TopKWindow}
_CRITERIA = {"auto": AutoCluster, "max": MaxCluster, "min": MinCluster}


def _parse_nonlinear(tensor_path: str) -> str:
    parts = [p for p in re.split(r"[\\/]", tensor_path) if p]
    for seg in parts:
        if seg.startswith("pre_"):
            return seg[len("pre_"):]
    return parts[-5] if len(parts) >= 5 else ""


class WindowSizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.window_size = cfg.window_size
        self.calibration = cfg.window_calibration

        if cfg.window_strategy not in _STRATEGIES:
            raise ValueError(f"unknown window_strategy {cfg.window_strategy!r}; expected "
                             f"one of {sorted(_STRATEGIES)}. Perplexity-guided per-layer "
                             f"search lives in profiling_api.search, not here -- this "
                             f"family only places a window from a saved histogram.")
        if cfg.window_criterion not in _CRITERIA:
            raise ValueError(f"unknown window_criterion {cfg.window_criterion!r}; "
                             f"expected one of {sorted(_CRITERIA)}")

        self.strategy = _STRATEGIES[cfg.window_strategy]()
        self.criterion = _CRITERIA[cfg.window_criterion]()

    def _size_tensor(self, tensor_path: str, window_size: int = None):
        if self.calibration == "Llama2_7B":
            return profile_tensor(tensor_path, window_size or self.window_size)
        return asdict(self._generic(tensor_path, window_size or self.window_size))

    def _generic(self, tensor_path: str, window_size: int) -> WindowSpec:
        nonlinear = _parse_nonlinear(tensor_path)

        tensor = torch.load(tensor_path, map_location="cpu")
        tensor = analysis_window(tensor)
        tensor = normalize_window(tensor, tensor_path)

        min_idx_raw, max_idx_raw = self.strategy.place(tensor, nonlinear)
        topk_tensor = tensor[min_idx_raw:max_idx_raw + 1]
        max_exp = index_to_exp(max_idx_raw)
        min_exp = index_to_exp(min_idx_raw)

        argmax_value = index_to_exp(tensor.argmax().item())

        window_sums = torch.tensor(
            [tensor[i:i + window_size].sum() for i in range(len(tensor) - window_size + 1)]
        )
        max_sum_pos = window_sums.argmax().item()
        tensor_windowed = tensor[max_sum_pos:max_sum_pos + window_size]

        max_value_idx = index_to_exp(max_sum_pos + (window_size - 1))
        min_value_idx = index_to_exp(max_sum_pos)

        indices = torch.arange(len(tensor_windowed))
        centroid = (tensor_windowed * indices).sum() / tensor_windowed.sum()

        max_i = exp_to_index(max_value_idx)
        min_i = exp_to_index(min_value_idx)
        max_sum = torch.sum(tensor[max_i - window_size:max_i]).item()
        min_sum = torch.sum(tensor[min_i:min_i + window_size]).item()

        return WindowSpec(
            argmax_value=argmax_value,
            tensor=tensor_windowed.tolist(),
            max_exp=max_exp,
            min_exp=min_exp,
            max_exp_unclamped=max_value_idx,
            min_exp_unclamped=min_value_idx,
            mean=topk_tensor.mean().item(),
            median=topk_tensor.median().item(),
            centroid=centroid.item(),
            cluster=self.criterion.cluster(max_sum, min_sum),
        )

    def size(self, store) -> None:
        for leaf in store.tensor_dirs():
            parts = re.split(r"[\\/]", leaf)
            if "exp_dist" not in parts:
                continue
            if not any(p.startswith("pre_") for p in parts):
                continue

            nonlinear_op = _parse_nonlinear(leaf)
            tail = parts[parts.index("exp_dist") + 1:]
            tensor_path = os.path.join(leaf, get_seq_len(leaf))
            data_dict = self._size_tensor(tensor_path)

            save_dir = os.path.join("distribution", store.model_name, nonlinear_op, *tail)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "profile.yaml"), "w") as f:
                yaml.dump(data_dict, f)

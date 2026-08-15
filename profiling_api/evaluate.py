from __future__ import annotations

import gc
import hashlib
import math
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Iterator, List, Optional

import torch

from profiling_api.apply import apply_assignment, set_profiling
from profiling_api.windows import ATTENTION, FFN, FfnWindow, SoftmaxWindow, WindowAssignment


@dataclass
class EvalResult:
    ppl: float
    ppl_token_weighted: float
    total_loss: float
    num_batches: int
    n_tokens: int
    wall_s: float
    gpu_s: Optional[float]
    apply_ms: float
    peak_mem_bytes: dict
    assignment_hash: str
    input_hash: str
    provenance: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop('peak_mem_bytes')
        row.pop('provenance')
        for device, peak in self.peak_mem_bytes.items():
            row[f'peak_mem_{device}'] = peak
        row.update({f'prov_{k}': v for k, v in self.provenance.items()})
        return row


@dataclass(frozen=True)
class LoadTiming:
    load_s: float
    device: str
    model_id: str = ''
    n_params: Optional[int] = None
    stages: dict = field(default_factory=dict)

    def with_model(self, model) -> 'LoadTiming':
        return replace(self, n_params=count_parameters(model))

    def with_stages(self, stages: dict) -> 'LoadTiming':
        return replace(self, stages=dict(stages or {}))

    @property
    def weights_s(self) -> Optional[float]:
        return self.stages.get('load_model')

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop('stages')
        row.update({f'load_{k}_s': v for k, v in self.stages.items()})
        return row


def count_parameters(model) -> Optional[int]:
    inner = getattr(model, 'model', model)
    parameters = getattr(inner, 'parameters', None)
    if parameters is None:
        return None
    return sum(p.numel() for p in parameters())


@contextmanager
def timed_load(model_id: str = '') -> Iterator[List[LoadTiming]]:
    box: List[LoadTiming] = []
    _sync()
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        _sync()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        box.append(LoadTiming(load_s=time.perf_counter() - t0, device=device,
                              model_id=model_id))


def hash_batches(batches) -> str:
    h = hashlib.sha256()
    for batch in batches:
        for ex in batch:
            ids = ex['input_ids']
            h.update(str(tuple(ids.shape)).encode())
            h.update(ids.detach().cpu().to(torch.int64).numpy().tobytes())
    return h.hexdigest()[:16]


def count_tokens(batches) -> int:
    return sum(int(ex['attention_mask'].sum()) for batch in batches for ex in batch)


def _peak_memory() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {f'cuda:{i}': torch.cuda.max_memory_allocated(i)
            for i in range(torch.cuda.device_count())}


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class WindowEvalHarness:
    def __init__(self, host, function_name: str = 'vlp', profiling: str = 'off',
                 seed_attention: SoftmaxWindow = None, seed_ffn: FfnWindow = None):
        self.host = host
        self.function_name = function_name
        self.profiling = profiling
        self.seed_attention = seed_attention or SoftmaxWindow(exp_dim=16, anchor=1)
        self.seed_ffn = seed_ffn or FfnWindow(exp_dim=16, pos_anchor=2, neg_anchor=2)

        self.baseline_ppl: Optional[float] = None
        self.baseline_result: Optional[EvalResult] = None
        self.input_hash: Optional[str] = None
        self.n_tokens: int = 0
        self.results: List[EvalResult] = []
        self._is_set_up = False

    @property
    def n_layers(self) -> int:
        return len(self.host.nonlinear_sites)

    def uniform(self, attention: SoftmaxWindow = None, ffn: FfnWindow = None) -> WindowAssignment:
        return WindowAssignment.uniform(
            self.n_layers,
            attention=attention if attention is not None else self.seed_attention,
            ffn=ffn if ffn is not None else self.seed_ffn,
        )

    def setup(self, measure_baseline: bool = True) -> None:
        if self._is_set_up:
            raise RuntimeError("setup() already ran; the model is patched irreversibly")

        if getattr(self.host.model, 'training', False):
            raise RuntimeError(
                "model is in training mode; dropout would make every eval nondeterministic. "
                "Call model.eval() before setup()."
            )

        self._require_eager_attention()

        self.input_hash = hash_batches(self.host.inputs)
        self.n_tokens = count_tokens(self.host.inputs)

        if measure_baseline:
            self.baseline_result = self._measure(apply_ms=0.0,
                                                 assignment_hash='baseline-unpatched')
            self.baseline_ppl = self.baseline_result.ppl

        self.host.patch_model(
            self.function_name,
            attention_parameters=self.seed_attention.to_kwargs(),
            ffn_parameters=self.seed_ffn.to_kwargs(),
            patch_attention=True,
            patch_ffn=True,
            run_eval=False,
        )

        if self.profiling == 'off':
            set_profiling(self.host, False)

        self._is_set_up = True

    def _require_eager_attention(self) -> None:
        config = getattr(self.host.model, 'config', None)
        impl = getattr(config, '_attn_implementation', None)
        if impl is not None and impl != 'eager':
            raise RuntimeError(
                f"model was loaded with attn_implementation={impl!r}, but the patched "
                f"attention forward only routes through the LUT when it is 'eager' "
                f"(custom_forward.llama_forward falls back to ALL_ATTENTION_FUNCTIONS "
                f"otherwise). Every softmax window would be silently ignored and the "
                f"perplexities would look plausible but measure nothing. Reload with "
                f"from_pretrained(..., attn_implementation='eager')."
            )

    def evaluate(self, assignment: WindowAssignment, record: bool = True) -> EvalResult:
        if not self._is_set_up:
            raise RuntimeError("call setup() before evaluate()")

        t0 = time.perf_counter()
        apply_assignment(self.host, assignment)
        apply_ms = (time.perf_counter() - t0) * 1000.0

        result = self._measure(apply_ms=apply_ms, assignment_hash=assignment.digest())
        if record:
            self.results.append(result)
        return result

    def _measure(self, apply_ms: float, assignment_hash: str) -> EvalResult:
        _reset_peak_memory()
        gc.collect()

        start_event = end_event = None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        _sync()
        t0 = time.perf_counter()
        self.host.run_inference()
        _sync()
        wall_s = time.perf_counter() - t0

        gpu_s = None
        if start_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            gpu_s = start_event.elapsed_time(end_event) / 1000.0

        total_loss = self.host.total_loss
        num_batches = self.host.num_batches
        mean_loss = total_loss / num_batches if num_batches else float('nan')

        return EvalResult(
            ppl=self.host.metric,
            ppl_token_weighted=math.exp(mean_loss) if num_batches else float('nan'),
            total_loss=total_loss,
            num_batches=num_batches,
            n_tokens=self.n_tokens,
            wall_s=wall_s,
            gpu_s=gpu_s,
            apply_ms=apply_ms,
            peak_mem_bytes=_peak_memory(),
            assignment_hash=assignment_hash,
            input_hash=self.input_hash,
            provenance=self._provenance(),
        )

    def _provenance(self) -> dict:
        import transformers

        return {
            'torch': str(torch.__version__),
            'transformers': str(transformers.__version__),
            'python': platform.python_version(),
            'cuda': str(torch.version.cuda or 'cpu'),
            'n_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'function_name': str(self.function_name),
            'profiling': bool(self.profiling),
            'model': str(getattr(self.host, 'model_name', '')),
        }

    def to_dataframe(self):
        import pandas as pd

        rows = [r.to_row() for r in self.results]
        if self.baseline_result is not None:
            rows.insert(0, self.baseline_result.to_row())
        return pd.DataFrame(rows)

    def write_csv(self, path: str) -> str:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path

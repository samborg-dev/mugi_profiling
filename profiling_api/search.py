from __future__ import annotations

import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from profiling_api.evaluate import WindowEvalHarness
from profiling_api.seed import HistogramSeeder, LayerSeed, fallback_assignment, seeds_to_rows
from profiling_api.windows import (ATTENTION, FfnWindow, SoftmaxWindow, WindowAssignment)

DEFAULT_MAX_ANCHORS = (0, 1, 2, 3, 4, 5, 6, 7)
DEFAULT_MIN_ANCHORS = (-7, -6, -5, -4, -3, -2, -1, 0)

FINAL_EVAL_RESERVE = 1


class StopReason(str, Enum):
    EXHAUSTED = "exhausted"
    NOISE_FLOOR = "noise_floor"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_EXHAUSTED = "time_exhausted"
    DIVERGED = "diverged"
    NOT_STARTED = "not_started"


@dataclass(frozen=True)
class CandidateGrid:
    max_anchors: Tuple[int, ...] = DEFAULT_MAX_ANCHORS
    min_anchors: Tuple[int, ...] = DEFAULT_MIN_ANCHORS
    radius: Optional[int] = None
    exp_dims: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.exp_dims is None:
            return

        dims = tuple(int(d) for d in self.exp_dims)
        bad = [d for d in dims if d < 1]
        if bad:
            raise ValueError(f"exp_dims must all be >= 1, got {bad}")
        object.__setattr__(self, 'exp_dims', dims)

    def candidates_for(self, seed: SoftmaxWindow) -> Tuple[SoftmaxWindow, ...]:
        pairs = {(anchor, "max") for anchor in self.max_anchors}
        pairs |= {(anchor, "min") for anchor in self.min_anchors}
        pairs.add((seed.anchor, seed.anchor_side))

        if self.radius is not None:
            pairs = {p for p in pairs
                     if abs(p[0] - seed.anchor) <= self.radius
                     or p == (seed.anchor, seed.anchor_side)}

        dims = tuple(self.exp_dims or ()) + (seed.exp_dim,)
        dims = tuple(dict.fromkeys(dims))

        seed_key = (seed.anchor, seed.anchor_side, seed.exp_dim)
        combos = {(anchor, side, dim) for (anchor, side) in pairs for dim in dims}
        combos.add(seed_key)

        ordered = sorted(combos, key=lambda c: (c != seed_key,
                                                abs(c[0] - seed.anchor)
                                                + abs(c[2] - seed.exp_dim),
                                                abs(c[0] - seed.anchor),
                                                c[1] != seed.anchor_side,
                                                c[0],
                                                c[2]))
        return tuple(SoftmaxWindow(exp_dim=dim, anchor=anchor, anchor_side=side,
                                   group_size=seed.group_size)
                     for anchor, side, dim in ordered)


@dataclass(frozen=True)
class SearchBudget:
    max_evals: Optional[int] = None
    max_evals_per_layer: Optional[int] = None
    repeats: int = 1
    noise_floor: float = 0.0
    patience: int = 3
    time_budget_s: Optional[float] = None

    def __post_init__(self):
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {self.repeats}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.noise_floor < 0:
            raise ValueError(f"noise_floor must be >= 0, got {self.noise_floor}")

    @property
    def evals_per_candidate(self) -> int:
        return self.repeats


@dataclass(frozen=True)
class CandidateRecord:
    layer: int
    site: str
    order: int
    anchor: int
    anchor_side: str
    exp_dim: int
    group_size: int
    ppl: float
    ppl_repeats: Tuple[float, ...]
    n_evals: int
    diverged: bool
    accepted: bool
    is_seed: bool
    delta_vs_incumbent: float
    wall_s: float
    apply_ms: float
    assignment_hash: str

    def to_row(self) -> dict:
        row = asdict(self)
        row['ppl_repeats'] = ';'.join(repr(p) for p in self.ppl_repeats)
        return row


@dataclass(frozen=True)
class LayerOutcome:
    layer: int
    site: str
    seed: LayerSeed
    chosen: SoftmaxWindow
    seed_ppl: float
    best_ppl: float
    n_candidates: int
    n_evals: int
    wall_s: float
    stop_reason: StopReason
    records: Tuple[CandidateRecord, ...] = ()

    @property
    def improvement(self) -> float:
        if not (math.isfinite(self.seed_ppl) and math.isfinite(self.best_ppl)):
            return 0.0
        return self.seed_ppl - self.best_ppl

    @property
    def changed(self) -> bool:
        return self.chosen != self.seed.window

    def to_row(self) -> dict:
        return {
            'layer': self.layer,
            'site': self.site,
            'seed_source': self.seed.source,
            'seed_anchor': self.seed.window.anchor,
            'seed_anchor_side': self.seed.window.anchor_side,
            'chosen_anchor': self.chosen.anchor,
            'chosen_anchor_side': self.chosen.anchor_side,
            'exp_dim': self.chosen.exp_dim,
            'group_size': self.chosen.group_size,
            'seed_ppl': self.seed_ppl,
            'best_ppl': self.best_ppl,
            'improvement': self.improvement,
            'changed': self.changed,
            'n_candidates': self.n_candidates,
            'n_evals': self.n_evals,
            'wall_s': self.wall_s,
            'stop_reason': self.stop_reason.value,
        }


@dataclass
class SearchResult:
    assignment: WindowAssignment
    outcomes: List[LayerOutcome]
    seeds: List[LayerSeed]
    budget: SearchBudget
    baseline_ppl: Optional[float] = None
    seed_ppl: Optional[float] = None
    final_ppl: Optional[float] = None
    n_evals: int = 0
    wall_s: float = 0.0
    eval_s_median: Optional[float] = None
    load_s: Optional[float] = None
    stopped_early: bool = False

    @property
    def improvement(self) -> float:
        if self.seed_ppl is None or self.final_ppl is None:
            return 0.0
        if not (math.isfinite(self.seed_ppl) and math.isfinite(self.final_ppl)):
            return 0.0
        return self.seed_ppl - self.final_ppl

    @property
    def n_layers_changed(self) -> int:
        return sum(1 for o in self.outcomes if o.changed)

    def records(self) -> List[CandidateRecord]:
        return [r for outcome in self.outcomes for r in outcome.records]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([r.to_row() for r in self.records()])

    def layer_dataframe(self):
        import pandas as pd

        return pd.DataFrame([o.to_row() for o in self.outcomes])

    def write_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

        layer_path = os.path.splitext(path)[0] + '_layers.csv'
        self.layer_dataframe().to_csv(layer_path, index=False)

        curve_path = os.path.splitext(path)[0] + '_locking_curve.csv'
        import pandas as pd

        pd.DataFrame(self.locking_curve()).to_csv(curve_path, index=False)
        return path

    def savings_report(self, n_layers: int = None) -> dict:
        n_layers = int(n_layers if n_layers is not None else len(self.outcomes))
        load_s = self.load_s
        eval_s = self.eval_s_median

        if load_s is None or eval_s is None:
            return {'n_layers': n_layers, 'load_s': load_s, 'eval_s_median': eval_s,
                    'n_evals': self.n_evals}

        search_s = self.n_evals * eval_s
        reloading_s = n_layers * load_s + search_s
        resident_s = load_s + search_s

        return {
            'n_layers': n_layers,
            'n_evals': self.n_evals,
            'load_s': load_s,
            'eval_s_median': eval_s,
            'reloading_s': reloading_s,
            'resident_s': resident_s,
            'saved_s': (n_layers - 1) * load_s,
            'speedup': reloading_s / resident_s if resident_s else float('nan'),
            'load_fraction_reloading': (n_layers * load_s) / reloading_s if reloading_s else float('nan'),
        }

    def summary(self) -> dict:
        return {
            'baseline_ppl': self.baseline_ppl,
            'seed_ppl': self.seed_ppl,
            'final_ppl': self.final_ppl,
            'improvement': self.improvement,
            'n_evals': self.n_evals,
            'n_layers': len(self.outcomes),
            'n_layers_changed': self.n_layers_changed,
            'wall_s': self.wall_s,
            'eval_s_median': self.eval_s_median,
            'load_s': self.load_s,
            'stopped_early': self.stopped_early,
            'assignment_digest': self.assignment.digest(),
            'budget': asdict(self.budget),
            'stop_reasons': {o.layer: o.stop_reason.value for o in self.outcomes},
        }

    def locking_curve(self) -> List[dict]:
        start = self.seed_ppl
        if start is None:
            start = self.outcomes[0].seed_ppl if self.outcomes else float('nan')

        rows = [{
            'n_locked': 0,
            'layer': None,
            'ppl': start,
            'improvement_vs_seed': 0.0,
            'step': 0.0,
            'changed': False,
            'cumulative_evals': 0,
            'cumulative_wall_s': 0.0,
            'stop_reason': None,
        }]

        evals = 0
        wall = 0.0
        previous = start
        for n, outcome in enumerate(self.outcomes, start=1):
            evals += outcome.n_evals
            wall += outcome.wall_s
            ppl = outcome.best_ppl
            step = (previous - ppl) if (math.isfinite(previous)
                                        and math.isfinite(ppl)) else 0.0
            rows.append({
                'n_locked': n,
                'layer': outcome.layer,
                'ppl': ppl,
                'improvement_vs_seed': (start - ppl) if (math.isfinite(start)
                                                         and math.isfinite(ppl)) else 0.0,
                'step': step,
                'changed': outcome.changed,
                'cumulative_evals': evals,
                'cumulative_wall_s': wall,
                'stop_reason': outcome.stop_reason.value,
            })
            previous = ppl

        return rows

    def write_yaml(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump({'summary': self.summary(),
                            'savings': self.savings_report(),
                            'seeds': seeds_to_rows(self.seeds),
                            'layers': [o.to_row() for o in self.outcomes],
                            'locking_curve': self.locking_curve()},
                           f, sort_keys=True)
        return path


class ProgressiveLayerSearch:
    def __init__(self, harness: WindowEvalHarness, seeder: HistogramSeeder = None,
                 budget: SearchBudget = None, grid: CandidateGrid = None,
                 site: str = ATTENTION, ffn: FfnWindow = None,
                 layers: Sequence[int] = None, measure_seed: bool = True,
                 progress: Callable[[str], None] = print):
        if not getattr(harness, '_is_set_up', False):
            raise RuntimeError(
                "harness.setup() has not run, so the model is unpatched and every "
                "reset_lut would fail on the first candidate. Call setup() first."
            )

        self.harness = harness
        self.seeder = seeder
        self.budget = budget or SearchBudget()
        self.grid = grid or CandidateGrid()
        self.site = site
        self.ffn = ffn
        self.measure_seed = measure_seed
        self.progress = progress or (lambda _: None)

        self.n_layers = harness.n_layers
        self.layers = ([int(l) for l in layers] if layers is not None
                       else list(range(self.n_layers)))

        unknown = [l for l in self.layers if not 0 <= l < self.n_layers]
        if unknown:
            raise ValueError(
                f"layers {unknown} are outside the patched model's range "
                f"0..{self.n_layers - 1}"
            )

        self.n_evals = 0
        self._t0 = None
        self._wall = []

    def seed_assignment(self) -> Tuple[WindowAssignment, List[LayerSeed]]:
        ffn = self.ffn or self.harness.seed_ffn
        if self.seeder is None:
            return fallback_assignment(self.n_layers,
                                       attention=self.harness.seed_attention, ffn=ffn)
        return self.seeder.seed_assignment(self.n_layers, ffn=ffn)

    def run(self) -> SearchResult:
        self._t0 = time.perf_counter()
        self.n_evals = 0
        self._wall = []

        assignment, seeds = self.seed_assignment()
        by_layer = {s.layer: s for s in seeds}

        n_histogram = sum(1 for s in seeds if s.from_histogram)
        self.progress(f"seeded {n_histogram}/{len(seeds)} layers from histograms, "
                      f"{len(seeds) - n_histogram} from fallback")

        incumbent = float('inf')
        seed_ppl = None
        if self.measure_seed:
            seed_ppl = self._evaluate(assignment)[0]
            incumbent = seed_ppl if math.isfinite(seed_ppl) else float('inf')
            self.progress(f"seed assignment ppl={seed_ppl:.6f}")

        frozen = assignment
        outcomes = []
        stopped_early = False

        for layer in self.layers:
            halt = self._halt_reason()
            if halt is not None:
                stopped_early = True
                seed = by_layer[layer]
                outcomes.append(LayerOutcome(
                    layer=layer, site=self.site, seed=seed,
                    chosen=frozen.get(layer, self.site), seed_ppl=incumbent,
                    best_ppl=incumbent, n_candidates=0, n_evals=0, wall_s=0.0,
                    stop_reason=halt))
                continue

            outcome = self._run_layer(layer, frozen, incumbent, by_layer[layer])
            frozen = frozen.with_override(layer, self.site, outcome.chosen)
            if math.isfinite(outcome.best_ppl):
                incumbent = min(incumbent, outcome.best_ppl)
            outcomes.append(outcome)

            self.progress(
                f"layer {layer:>3}: {outcome.seed.window.anchor_side}"
                f"{outcome.seed.window.anchor:+d} -> {outcome.chosen.anchor_side}"
                f"{outcome.chosen.anchor:+d}  ppl={outcome.best_ppl:.6f} "
                f"gain={outcome.improvement:.6f}  evals={outcome.n_evals} "
                f"stop={outcome.stop_reason.value}")

            if outcome.stop_reason in (StopReason.BUDGET_EXHAUSTED,
                                       StopReason.TIME_EXHAUSTED):
                stopped_early = True

        final_ppl = self._evaluate(frozen)[0]

        return SearchResult(
            assignment=frozen, outcomes=outcomes, seeds=seeds, budget=self.budget,
            baseline_ppl=self.harness.baseline_ppl, seed_ppl=seed_ppl,
            final_ppl=final_ppl, n_evals=self.n_evals,
            wall_s=time.perf_counter() - self._t0,
            eval_s_median=statistics.median(self._wall) if self._wall else None,
            stopped_early=stopped_early)

    def _run_layer(self, layer: int, frozen: WindowAssignment, incumbent: float,
                   seed: LayerSeed) -> LayerOutcome:
        t0 = time.perf_counter()
        seed_window = frozen.get(layer, self.site)
        candidates = self.grid.candidates_for(seed_window)

        best_window = seed_window
        best_ppl = incumbent
        misses = 0
        layer_evals = 0
        stop_reason = StopReason.EXHAUSTED

        records = [CandidateRecord(
            layer=layer, site=self.site, order=0, anchor=seed_window.anchor,
            anchor_side=seed_window.anchor_side, exp_dim=seed_window.exp_dim,
            group_size=seed_window.group_size, ppl=incumbent, ppl_repeats=(incumbent,),
            n_evals=0, diverged=not math.isfinite(incumbent), accepted=True,
            is_seed=True, delta_vs_incumbent=0.0, wall_s=0.0, apply_ms=0.0,
            assignment_hash=frozen.digest())]

        for order, candidate in enumerate(candidates[1:], start=1):
            charge = self.budget.evals_per_candidate

            if not self._can_charge(charge):
                stop_reason = StopReason.BUDGET_EXHAUSTED
                break
            if (self.budget.max_evals_per_layer is not None
                    and layer_evals + charge > self.budget.max_evals_per_layer):
                stop_reason = StopReason.BUDGET_EXHAUSTED
                break
            if self._out_of_time():
                stop_reason = StopReason.TIME_EXHAUSTED
                break

            trial = frozen.with_override(layer, self.site, candidate)
            score, ppls, wall_s, apply_ms = self._evaluate_candidate(trial)
            layer_evals += charge

            previous = best_ppl
            accepted = (best_ppl - score) > self.budget.noise_floor
            if accepted:
                best_window, best_ppl, misses = candidate, score, 0
            else:
                misses += 1

            records.append(CandidateRecord(
                layer=layer, site=self.site, order=order, anchor=candidate.anchor,
                anchor_side=candidate.anchor_side, exp_dim=candidate.exp_dim,
                group_size=candidate.group_size, ppl=score, ppl_repeats=ppls,
                n_evals=charge, diverged=not math.isfinite(score), accepted=accepted,
                is_seed=False, delta_vs_incumbent=previous - score,
                wall_s=wall_s, apply_ms=apply_ms, assignment_hash=trial.digest()))

            if misses >= self.budget.patience:
                stop_reason = StopReason.NOISE_FLOOR
                break

        if not math.isfinite(best_ppl):
            stop_reason = StopReason.DIVERGED

        return LayerOutcome(
            layer=layer, site=self.site, seed=seed, chosen=best_window,
            seed_ppl=incumbent, best_ppl=best_ppl, n_candidates=len(records) - 1,
            n_evals=layer_evals, wall_s=time.perf_counter() - t0,
            stop_reason=stop_reason, records=tuple(records))

    def _evaluate_candidate(self, assignment: WindowAssignment
                            ) -> Tuple[float, Tuple[float, ...], float, float]:
        ppls, walls, applies = [], [], []
        for _ in range(self.budget.repeats):
            ppl, wall_s, apply_ms = self._evaluate(assignment)
            ppls.append(ppl)
            walls.append(wall_s)
            applies.append(apply_ms)

        finite = [p for p in ppls if math.isfinite(p)]
        score = statistics.fmean(finite) if finite else float('inf')
        return score, tuple(ppls), sum(walls), statistics.median(applies)

    def _evaluate(self, assignment: WindowAssignment) -> Tuple[float, float, float]:
        result = self.harness.evaluate(assignment)
        self.n_evals += 1
        self._wall.append(result.wall_s)
        return result.ppl, result.wall_s, result.apply_ms

    def _can_charge(self, charge: int) -> bool:
        if self.budget.max_evals is None:
            return True
        return self.n_evals + charge + FINAL_EVAL_RESERVE <= self.budget.max_evals

    def _out_of_time(self) -> bool:
        return (self.budget.time_budget_s is not None
                and (time.perf_counter() - self._t0) > self.budget.time_budget_s)

    def _halt_reason(self) -> Optional[StopReason]:
        if self._out_of_time():
            return StopReason.TIME_EXHAUSTED
        if not self._can_charge(self.budget.evals_per_candidate):
            return StopReason.BUDGET_EXHAUSTED
        return None

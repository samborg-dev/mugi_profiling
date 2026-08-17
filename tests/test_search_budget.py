import math
from dataclasses import dataclass

import pytest

from profiling_api.search import (CandidateGrid, LayerOutcome, ProgressiveLayerSearch,
                                  SearchBudget, StopReason, paired_ppl_ci)
from profiling_api.windows import ATTENTION, FFN, FfnWindow, SoftmaxWindow, WindowAssignment

SEED_ANCHOR = 2
TARGETS = (0, 3, 5, 1)


@dataclass
class FakeResult:
    ppl: float
    wall_s: float = 0.5
    apply_ms: float = 1.0


class FakeHarness:
    def __init__(self, targets=TARGETS, base=6.0, scale=0.01, nan_at=(),
                 exp_dim=16, group_size=256):
        self.targets = tuple(targets)
        self.base = base
        self.scale = scale
        self.nan_at = set(nan_at)
        self.n_layers = len(self.targets)
        self.seed_attention = SoftmaxWindow(exp_dim=exp_dim, anchor=SEED_ANCHOR,
                                            anchor_side='max', group_size=group_size)
        self.seed_ffn = FfnWindow(exp_dim=exp_dim, pos_anchor=2, neg_anchor=2,
                                  group_size=group_size)
        self.baseline_ppl = 5.75
        self.results = []
        self.seen = []
        self._is_set_up = True

    def cost(self, window):
        return abs(window.anchor - 0) * self.scale

    def evaluate(self, assignment, record=True):
        self.seen.append(assignment)
        total = 0.0
        for layer, target in enumerate(self.targets):
            window = assignment.get(layer, ATTENTION)
            if (layer, window.anchor, window.anchor_side) in self.nan_at:
                result = FakeResult(float('nan'))
                self.results.append(result)
                return result
            penalty = 0 if window.anchor_side == 'max' else 1
            total += (abs(window.anchor - target) + penalty) * self.scale
        result = FakeResult(self.base + total)
        self.results.append(result)
        return result


def search(harness, progress=None, **budget_kwargs):
    budget_kwargs.setdefault('patience', 99)
    return ProgressiveLayerSearch(harness, budget=SearchBudget(**budget_kwargs),
                                  progress=progress or (lambda _: None))


def anchors(assignment, n_layers):
    return [assignment.get(layer, ATTENTION).anchor for layer in range(n_layers)]


class TestConvergence:
    def test_finds_the_known_optimum_for_every_layer(self):
        harness = FakeHarness()
        result = search(harness, noise_floor=0.0).run()
        assert tuple(anchors(result.assignment, harness.n_layers)) == TARGETS
        assert all(result.assignment.get(l, ATTENTION).anchor_side == 'max'
                   for l in range(harness.n_layers))

    def test_the_default_patience_still_reaches_the_optimum(self):
        harness = FakeHarness()
        result = search(harness, patience=3).run()
        assert tuple(anchors(result.assignment, harness.n_layers)) == TARGETS

    def test_early_termination_costs_far_fewer_evals_than_exhausting_the_grid(self):
        cheap = search(FakeHarness(), patience=3).run()
        thorough = search(FakeHarness(), patience=99).run()
        assert cheap.n_evals < thorough.n_evals
        assert cheap.assignment.digest() == thorough.assignment.digest()

    def test_final_ppl_never_regresses_past_the_seed(self):
        result = search(FakeHarness(), noise_floor=0.0).run()
        assert result.final_ppl <= result.seed_ppl
        assert result.improvement > 0

    def test_every_layer_is_assigned_both_sites(self):
        harness = FakeHarness()
        result = search(harness).run()
        for layer in range(harness.n_layers):
            assert result.assignment.get(layer, ATTENTION) is not None
            assert result.assignment.get(layer, FFN) is not None

    def test_a_layer_subset_leaves_the_others_at_seed(self):
        harness = FakeHarness()
        result = ProgressiveLayerSearch(harness, budget=SearchBudget(patience=99),
                                        layers=[1], progress=lambda _: None).run()
        assert result.assignment.get(1, ATTENTION).anchor == TARGETS[1]
        for layer in (0, 2, 3):
            assert result.assignment.get(layer, ATTENTION).anchor == SEED_ANCHOR

    def test_layers_outside_the_model_are_refused(self):
        with pytest.raises(ValueError, match="outside the patched model"):
            ProgressiveLayerSearch(FakeHarness(), layers=[0, 99])


class TestNoiseFloor:
    def test_a_floor_above_every_gap_keeps_every_seed(self):
        harness = FakeHarness(scale=0.005)
        result = search(harness, noise_floor=0.02).run()

        assert result.n_layers_changed == 0
        assert anchors(result.assignment, harness.n_layers) == [SEED_ANCHOR] * 4
        assert all(o.improvement == 0.0 for o in result.outcomes)

    def test_a_gap_exactly_at_the_floor_is_not_an_improvement(self):
        harness = FakeHarness(scale=0.01)
        result = search(harness, noise_floor=0.01).run()

        accepted = [r for r in result.records() if r.accepted and not r.is_seed]
        assert all(r.delta_vs_incumbent > 0.01 for r in accepted)

    def test_patience_stops_a_layer_before_the_grid_is_exhausted(self):
        harness = FakeHarness()
        result = search(harness, patience=2).run()

        assert any(o.stop_reason is StopReason.NOISE_FLOOR for o in result.outcomes)
        assert all(o.n_candidates < 15 for o in result.outcomes)

    def test_a_flat_landscape_terminates_at_the_floor_not_the_grid_end(self):
        harness = FakeHarness(scale=0.0)
        result = search(harness, patience=2, noise_floor=0.0).run()

        assert all(o.stop_reason is StopReason.NOISE_FLOOR for o in result.outcomes)
        assert result.n_layers_changed == 0


class TestBudget:
    @pytest.mark.parametrize('cap', [6, 10, 20, 40])
    def test_the_charge_count_never_exceeds_max_evals(self, cap):
        result = search(FakeHarness(), max_evals=cap).run()
        assert result.n_evals <= cap

    def test_exhausting_the_budget_still_returns_a_complete_assignment(self):
        harness = FakeHarness()
        result = search(harness, max_evals=10).run()

        assert result.stopped_early
        assert sorted({layer for layer, _, _ in result.assignment.items()}) == [0, 1, 2, 3]
        assert any(o.stop_reason is StopReason.BUDGET_EXHAUSTED for o in result.outcomes)

    def test_untouched_layers_keep_their_seed_window(self):
        harness = FakeHarness()
        result = search(harness, max_evals=10).run()

        starved = [o for o in result.outcomes
                   if o.stop_reason is StopReason.BUDGET_EXHAUSTED and o.n_evals == 0]
        assert starved
        for outcome in starved:
            assert outcome.chosen == outcome.seed.window

    def test_repeats_multiply_the_charge_per_candidate(self):
        result = search(FakeHarness(), repeats=3, max_evals=30).run()
        candidates = [r for r in result.records() if not r.is_seed]

        assert {r.n_evals for r in candidates} == {3}
        assert all(len(r.ppl_repeats) == 3 for r in candidates)

    def test_repeats_buy_fewer_candidates_under_the_same_budget(self):
        single = search(FakeHarness(), repeats=1, max_evals=30).run()
        tripled = search(FakeHarness(), repeats=3, max_evals=30).run()

        n_single = len([r for r in single.records() if not r.is_seed])
        n_tripled = len([r for r in tripled.records() if not r.is_seed])
        assert n_tripled < n_single

    def test_a_repeated_candidate_is_scored_by_its_mean(self):
        result = search(FakeHarness(), repeats=3, max_evals=30).run()
        record = next(r for r in result.records() if not r.is_seed)
        assert record.ppl == pytest.approx(sum(record.ppl_repeats) / 3)

    def test_per_layer_cap_bounds_each_layer(self):
        result = search(FakeHarness(), max_evals_per_layer=4).run()
        assert all(o.n_evals <= 4 for o in result.outcomes)

    def test_an_unset_up_harness_is_refused(self):
        harness = FakeHarness()
        harness._is_set_up = False
        with pytest.raises(RuntimeError, match="setup"):
            ProgressiveLayerSearch(harness)

    @pytest.mark.parametrize('kwargs', [{'repeats': 0}, {'patience': 0},
                                        {'noise_floor': -1.0}])
    def test_nonsense_budgets_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            SearchBudget(**kwargs)


class TestFreezing:
    def test_predecessors_are_frozen_and_successors_stay_at_seed(self):
        harness = FakeHarness()
        result = search(harness, noise_floor=0.0).run()

        chosen = {o.layer: o.chosen for o in result.outcomes}
        seeds = {s.layer: s.window for s in result.seeds}

        for outcome in result.outcomes:
            for record in outcome.records:
                if record.is_seed:
                    continue
                seen = next(a for a in harness.seen
                            if a.digest() == record.assignment_hash)
                for layer in range(harness.n_layers):
                    window = seen.get(layer, ATTENTION)
                    if layer < outcome.layer:
                        assert window == chosen[layer]
                    elif layer > outcome.layer:
                        assert window == seeds[layer]
                    else:
                        assert window.anchor == record.anchor

    def test_the_search_never_mutates_the_seed_assignment(self):
        harness = FakeHarness()
        searcher = search(harness, noise_floor=0.0)
        seed_assignment, _ = searcher.seed_assignment()
        before = seed_assignment.digest()
        searcher.run()
        assert seed_assignment.digest() == before


class TestDivergence:
    def test_a_diverged_candidate_is_recorded_and_never_chosen(self):
        harness = FakeHarness(nan_at=[(1, 3, 'max')])
        result = search(harness).run()

        diverged = [r for r in result.records() if r.diverged and not r.is_seed]
        assert diverged
        assert not any(r.accepted for r in diverged)
        assert result.assignment.get(1, ATTENTION).anchor != 3

    def test_divergence_does_not_end_the_sweep(self):
        harness = FakeHarness(nan_at=[(0, 1, 'max')])
        result = search(harness).run()

        layer0 = next(o for o in result.outcomes if o.layer == 0)
        assert layer0.n_candidates > 1
        assert result.assignment.get(0, ATTENTION).anchor == TARGETS[0]

    def test_a_layer_where_everything_diverges_falls_back_to_its_seed(self):
        every = [(2, anchor, side) for anchor in range(-7, 8)
                 for side in ('max', 'min')]
        harness = FakeHarness(nan_at=every)
        result = search(harness).run()

        layer2 = next(o for o in result.outcomes if o.layer == 2)
        assert layer2.stop_reason is StopReason.DIVERGED
        assert layer2.chosen == layer2.seed.window
        assert len(result.outcomes) == harness.n_layers

    def test_a_wiring_error_is_not_swallowed(self):
        harness = FakeHarness()

        def explode(assignment, record=True):
            raise KeyError("assignment names layer 99")

        harness.evaluate = explode
        with pytest.raises(KeyError):
            search(harness).run()


class TestCandidateGrid:
    def test_the_seed_comes_first(self):
        seed = SoftmaxWindow(exp_dim=16, anchor=2, anchor_side='max')
        candidates = CandidateGrid().candidates_for(seed)
        assert candidates[0] == seed

    def test_candidates_are_ordered_by_distance_from_the_seed(self):
        seed = SoftmaxWindow(exp_dim=16, anchor=2, anchor_side='max')
        distances = [abs(w.anchor - seed.anchor)
                     for w in CandidateGrid().candidates_for(seed)]
        assert distances == sorted(distances)

    def test_the_full_grid_matches_the_newton_candidate_count(self):
        seed = SoftmaxWindow(exp_dim=16, anchor=2, anchor_side='max')
        assert len(CandidateGrid().candidates_for(seed)) == 16

    def test_radius_bounds_the_candidate_set(self):
        seed = SoftmaxWindow(exp_dim=16, anchor=2, anchor_side='max')
        candidates = CandidateGrid(radius=1).candidates_for(seed)
        assert all(abs(w.anchor - seed.anchor) <= 1 for w in candidates)
        assert candidates[0] == seed

    def test_an_off_grid_seed_is_still_offered_as_a_candidate(self):
        seed = SoftmaxWindow(exp_dim=16, anchor=6, anchor_side='min')
        candidates = CandidateGrid().candidates_for(seed)
        assert candidates[0] == seed

    def test_candidates_inherit_the_seed_geometry(self):
        seed = SoftmaxWindow(exp_dim=11, anchor=2, anchor_side='max', group_size=128)
        for window in CandidateGrid().candidates_for(seed):
            assert window.exp_dim == 11 and window.group_size == 128


class TestWindowSizeSweep:
    def test_exp_dims_actually_vary_the_window_size(self):
        seed = SoftmaxWindow(exp_dim=10, anchor=2, anchor_side='max')
        candidates = CandidateGrid(exp_dims=(8, 10, 12)).candidates_for(seed)
        assert {w.exp_dim for w in candidates} == {8, 10, 12}

    def test_the_sweep_multiplies_the_grid(self):
        seed = SoftmaxWindow(exp_dim=10, anchor=2, anchor_side='max')
        assert len(CandidateGrid(exp_dims=(8, 10, 12)).candidates_for(seed)) == 16 * 3

    def test_the_seed_window_is_still_offered_first(self):
        seed = SoftmaxWindow(exp_dim=10, anchor=2, anchor_side='max')
        candidates = CandidateGrid(exp_dims=(8, 12, 16)).candidates_for(seed)
        assert candidates[0] == seed

    def test_the_seed_size_is_not_duplicated_when_it_is_already_listed(self):
        seed = SoftmaxWindow(exp_dim=10, anchor=2, anchor_side='max')
        candidates = CandidateGrid(exp_dims=(8, 10, 12)).candidates_for(seed)
        assert len(candidates) == len(set(candidates))

    def test_ordering_is_nearest_first_in_both_dimensions(self):
        seed = SoftmaxWindow(exp_dim=10, anchor=2, anchor_side='max')
        candidates = CandidateGrid(exp_dims=(8, 10, 12)).candidates_for(seed)
        distances = [abs(w.anchor - seed.anchor) + abs(w.exp_dim - seed.exp_dim)
                     for w in candidates]
        assert distances == sorted(distances)

    def test_group_size_is_still_inherited(self):
        seed = SoftmaxWindow(exp_dim=10, anchor=2, anchor_side='max', group_size=128)
        candidates = CandidateGrid(exp_dims=(8, 12)).candidates_for(seed)
        assert all(w.group_size == 128 for w in candidates)

    def test_no_sweep_reproduces_the_newton_grid_exactly(self):
        seed = SoftmaxWindow(exp_dim=16, anchor=2, anchor_side='max')
        assert (CandidateGrid(exp_dims=None).candidates_for(seed)
                == CandidateGrid().candidates_for(seed))

    def test_the_sweep_composes_with_radius(self):
        seed = SoftmaxWindow(exp_dim=10, anchor=2, anchor_side='max')
        candidates = CandidateGrid(radius=1, exp_dims=(8, 10)).candidates_for(seed)
        assert all(abs(w.anchor - seed.anchor) <= 1 for w in candidates)
        assert {w.exp_dim for w in candidates} == {8, 10}

    def test_a_window_size_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="exp_dims"):
            CandidateGrid(exp_dims=(10, 0))

    def test_the_search_can_choose_a_different_window_size(self):
        harness = FakeHarness(exp_dim=10)
        result = ProgressiveLayerSearch(
            harness, budget=SearchBudget(noise_floor=0.0, patience=99),
            grid=CandidateGrid(exp_dims=(8, 10, 12)),
            progress=lambda _: None).run()
        assert all(o.chosen.exp_dim in (8, 10, 12) for o in result.outcomes)


class TestLockingCurve:
    def test_the_curve_starts_at_the_measured_seed_and_covers_every_layer(self):
        harness = FakeHarness()
        result = search(harness).run()
        curve = result.locking_curve()

        assert len(curve) == harness.n_layers + 1
        assert curve[0]['n_locked'] == 0
        assert curve[0]['ppl'] == result.seed_ppl
        assert [row['n_locked'] for row in curve] == list(range(harness.n_layers + 1))

    def test_the_curve_never_goes_uphill(self):
        result = search(FakeHarness()).run()
        ppls = [row['ppl'] for row in result.locking_curve()]
        assert ppls == sorted(ppls, reverse=True)

    def test_the_curve_ends_at_the_final_perplexity(self):
        result = search(FakeHarness()).run()
        assert result.locking_curve()[-1]['ppl'] == pytest.approx(result.final_ppl)

    def test_cumulative_evaluations_match_the_search_total(self):
        result = search(FakeHarness()).run()
        curve = result.locking_curve()
        assert curve[-1]['cumulative_evals'] == sum(o.n_evals for o in result.outcomes)

    def test_the_steps_sum_to_the_total_improvement(self):
        result = search(FakeHarness()).run()
        curve = result.locking_curve()
        assert sum(row['step'] for row in curve) == pytest.approx(
            curve[-1]['improvement_vs_seed'])

    def test_the_curve_survives_a_layer_that_diverges(self):
        harness = FakeHarness(nan_at={(1, 3, 'max')})
        curve = search(harness).run().locking_curve()
        assert len(curve) == harness.n_layers + 1
        assert all(math.isfinite(row['ppl']) for row in curve)


class LossHarness(FakeHarness):
    def __init__(self, n_batches=64, spread=0.4, shift=0.02, **kwargs):
        super().__init__(**kwargs)
        self.n_batches = n_batches
        self.spread = spread
        self.shift = shift

    def evaluate(self, assignment, record=True):
        result = super().evaluate(assignment)
        offset = (result.ppl - self.base) * self.shift
        result.batch_losses = tuple(
            1.8 + self.spread * math.sin(i) + offset for i in range(self.n_batches))
        return result


class TestPairedAcceptance:
    def test_the_interval_is_none_without_enough_batches(self):
        assert paired_ppl_ci((1.8,), (1.7,)) is None
        assert paired_ppl_ci((), ()) is None

    def test_mismatched_batch_counts_are_refused(self):
        assert paired_ppl_ci((1.8, 1.9, 2.0), (1.7, 1.8)) is None

    def test_a_nan_loss_is_refused(self):
        assert paired_ppl_ci((1.8, float('nan'), 2.0), (1.7, 1.8, 1.9)) is None

    def test_identical_losses_give_an_interval_containing_zero(self):
        losses = tuple(1.8 + 0.3 * math.sin(i) for i in range(64))
        lo, hi = paired_ppl_ci(losses, losses, n_resamples=500)
        assert lo == 0.0 and hi == 0.0

    def test_a_uniformly_better_candidate_gives_a_positive_interval(self):
        inc = tuple(1.8 + 0.3 * math.sin(i) for i in range(64))
        cand = tuple(x - 0.05 for x in inc)
        lo, hi = paired_ppl_ci(inc, cand, n_resamples=500)
        assert lo > 0 and hi > 0

    def test_pairing_is_tighter_than_the_unpaired_spread(self):
        inc = tuple(1.8 + 0.6 * math.sin(i) for i in range(64))
        cand = tuple(x - 0.01 for x in inc)
        lo, hi = paired_ppl_ci(inc, cand, n_resamples=500)
        assert (hi - lo) < 0.5

    def test_paired_mode_falls_back_when_the_host_records_no_batches(self):
        harness = FakeHarness()
        result = ProgressiveLayerSearch(
            harness, budget=SearchBudget(paired=True, noise_floor=0.0, patience=99),
            progress=lambda _: None).run()
        assert result.final_ppl < result.seed_ppl
        assert all(r.paired_ci_low is None for r in result.records())

    def test_paired_mode_records_the_interval_when_batches_exist(self):
        harness = LossHarness()
        result = ProgressiveLayerSearch(
            harness, budget=SearchBudget(paired=True, noise_floor=0.0, patience=99),
            progress=lambda _: None).run()
        scored = [r for r in result.records() if not r.is_seed]
        assert any(r.paired_ci_low is not None for r in scored)

    def test_a_paired_win_still_covers_every_layer(self):
        harness = LossHarness()
        result = ProgressiveLayerSearch(
            harness, budget=SearchBudget(paired=True, noise_floor=0.0, patience=99),
            progress=lambda _: None).run()
        assert len(result.assignment.layers) == harness.n_layers

    def test_a_confidence_outside_the_open_interval_is_rejected(self):
        with pytest.raises(ValueError, match="paired_confidence"):
            SearchBudget(paired=True, paired_confidence=100.0)

    def test_non_positive_resamples_are_rejected(self):
        with pytest.raises(ValueError, match="paired_resamples"):
            SearchBudget(paired=True, paired_resamples=0)

    def test_the_default_search_is_unpaired(self):
        assert SearchBudget().paired is False


class TestReporting:
    def test_the_trace_covers_every_evaluated_candidate(self):
        harness = FakeHarness()
        result = search(harness).run()
        charged = sum(r.n_evals for r in result.records())
        assert result.n_evals == charged + 1 + 1

    def test_savings_are_entirely_the_reloads(self):
        result = search(FakeHarness()).run()
        result.load_s = 240.0
        savings = result.savings_report(n_layers=32)

        assert savings['saved_s'] == pytest.approx(31 * 240.0)
        assert savings['speedup'] > 1
        assert savings['reloading_s'] - savings['resident_s'] == pytest.approx(
            savings['saved_s'])

    def test_savings_degrade_gracefully_without_a_load_measurement(self):
        result = search(FakeHarness()).run()
        assert result.savings_report(32)['load_s'] is None

    def test_the_summary_records_the_stop_reason_of_every_layer(self):
        harness = FakeHarness()
        result = search(harness, max_evals=10).run()
        assert len(result.summary()['stop_reasons']) == harness.n_layers

    def test_a_candidate_row_flattens_its_repeats(self):
        result = search(FakeHarness(), repeats=2, max_evals=20).run()
        record = next(r for r in result.records() if not r.is_seed)
        assert isinstance(record.to_row()['ppl_repeats'], str)
        assert record.to_row()['ppl_repeats'].count(';') == 1

    def test_the_seed_record_costs_nothing(self):
        result = search(FakeHarness()).run()
        seeds = [r for r in result.records() if r.is_seed]
        assert len(seeds) == 4
        assert all(r.n_evals == 0 for r in seeds)

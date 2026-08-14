import math
import os

import pytest
import torch

from profiling_api.evaluate import WindowEvalHarness, count_parameters, timed_load
from profiling_api.search import (CandidateGrid, ProgressiveLayerSearch, SearchBudget,
                                  StopReason)
from profiling_api.seed import HistogramSeeder
from profiling_api.windows import ATTENTION, FFN, SoftmaxWindow, WindowAssignment
from tiny_host import TinyHost, harness, host, load_tiny  # noqa: F401

pd = pytest.importorskip("pandas")

NARROW = CandidateGrid(radius=1)


def fresh_harness(tmp_path):
    model = load_tiny()
    model.eval()
    h = WindowEvalHarness(TinyHost(model, tmp_path))
    h.setup()
    return h


def run(harness, **budget):
    budget.setdefault('patience', 99)
    return ProgressiveLayerSearch(harness, budget=SearchBudget(**budget),
                                  grid=NARROW, progress=lambda _: None).run()


class TestEndToEnd:
    def test_the_search_completes_and_assigns_every_layer(self, harness):
        result = run(harness)

        assert len(result.outcomes) == harness.n_layers
        for layer in range(harness.n_layers):
            assert isinstance(result.assignment.get(layer, ATTENTION), SoftmaxWindow)
            assert result.assignment.get(layer, FFN) is not None

    def test_the_model_is_never_reloaded(self, harness):
        model_id = id(harness.host.model)
        sites_id = id(harness.host.nonlinear_sites)

        run(harness)

        assert id(harness.host.model) == model_id
        assert id(harness.host.nonlinear_sites) == sites_id

    def test_no_profile_tensors_are_written_during_the_search(self, harness, tmp_path):
        run(harness)
        assert list(tmp_path.rglob('*.pt')) == []

    def test_the_trace_and_the_harness_log_agree(self, harness):
        result = run(harness)
        charged = sum(r.n_evals for r in result.records())

        assert result.n_evals == charged + 2
        assert len(harness.results) == result.n_evals

    def test_the_final_assignment_is_no_worse_than_the_seed(self, harness):
        result = run(harness)

        assert math.isfinite(result.seed_ppl)
        if math.isfinite(result.final_ppl):
            assert result.final_ppl <= result.seed_ppl + 1e-9

    def test_the_baseline_survives_the_search(self, harness):
        baseline = harness.baseline_ppl
        result = run(harness)
        assert result.baseline_ppl == baseline

    def test_two_identical_searches_agree_on_cpu(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        first = run(fresh_harness(tmp_path), noise_floor=0.0)
        second = run(fresh_harness(tmp_path), noise_floor=0.0)

        assert first.assignment.digest() == second.assignment.digest()
        assert first.final_ppl == pytest.approx(second.final_ppl)


class TestDivergenceOnTheRealKernel:
    def test_a_starving_window_is_recorded_but_never_chosen(self, harness):
        starving = SoftmaxWindow(exp_dim=1, anchor=-14, anchor_side='max')
        seeded = WindowAssignment()
        for layer in range(harness.n_layers):
            seeded.set(layer, ATTENTION, harness.seed_attention)
            seeded.set(layer, FFN, harness.seed_ffn)

        broken = seeded.with_override(0, ATTENTION, starving)
        result = harness.evaluate(broken)

        assert not math.isfinite(result.ppl) or result.ppl > harness.baseline_ppl

    def test_the_search_survives_a_grid_that_contains_a_bad_window(self, harness):
        wide = CandidateGrid(max_anchors=(-14, 0, 2), min_anchors=(-14, 0))
        result = ProgressiveLayerSearch(harness, budget=SearchBudget(patience=99),
                                        grid=wide, progress=lambda _: None).run()

        assert len(result.outcomes) == harness.n_layers
        assert all(o.stop_reason is not StopReason.NOT_STARTED for o in result.outcomes)


class TestArtifacts:
    def test_the_trace_csv_round_trips(self, harness, tmp_path):
        result = run(harness)
        path = result.write_csv(str(tmp_path / 'out' / 'trace.csv'))

        frame = pd.read_csv(path)
        assert len(frame) == len(result.records())
        assert set(frame['layer']) == set(range(harness.n_layers))

    def test_the_layer_csv_is_written_alongside(self, harness, tmp_path):
        result = run(harness)
        path = result.write_csv(str(tmp_path / 'out' / 'trace.csv'))
        layers = pd.read_csv(os.path.splitext(path)[0] + '_layers.csv')

        assert len(layers) == harness.n_layers
        assert 'stop_reason' in layers.columns

    def test_the_trace_joins_to_the_harness_log_on_assignment_hash(self, harness, tmp_path):
        result = run(harness)
        trace = pd.read_csv(result.write_csv(str(tmp_path / 'out' / 'trace.csv')))
        evals = pd.read_csv(harness.write_csv(str(tmp_path / 'out' / 'evals.csv')))

        assert set(trace['assignment_hash']) & set(evals['assignment_hash'])

    def test_the_assignment_yaml_round_trips(self, harness, tmp_path):
        result = run(harness)
        path = result.assignment.to_yaml(str(tmp_path / 'out' / 'assignment.yaml'))

        assert WindowAssignment.from_yaml(path).digest() == result.assignment.digest()

    def test_the_summary_yaml_is_written(self, harness, tmp_path):
        import yaml

        result = run(harness)
        path = result.write_yaml(str(tmp_path / 'out' / 'summary.yaml'))
        with open(path) as f:
            summary = yaml.safe_load(f)

        assert summary['summary']['n_layers'] == harness.n_layers
        assert len(summary['seeds']) == harness.n_layers
        assert 'savings' in summary


class TestSeedingAgainstTheHarness:
    def test_an_absent_profile_root_seeds_every_layer_from_the_fallback(self, harness, tmp_path):
        seeder = HistogramSeeder(str(tmp_path / 'no_profile'))
        search = ProgressiveLayerSearch(harness, seeder=seeder, grid=NARROW,
                                        budget=SearchBudget(patience=99),
                                        progress=lambda _: None)
        result = search.run()

        assert all(not s.from_histogram for s in result.seeds)
        assert len(result.seeds) == harness.n_layers

    def test_without_a_seeder_the_harness_seed_is_the_starting_point(self, harness):
        search = ProgressiveLayerSearch(harness, grid=NARROW,
                                        budget=SearchBudget(patience=99),
                                        progress=lambda _: None)
        assignment, seeds = search.seed_assignment()

        assert all(assignment.get(l, ATTENTION) == harness.seed_attention
                   for l in range(harness.n_layers))
        assert all(s.detail == "seeding disabled" for s in seeds)


class TestLoadTiming:
    def test_a_load_is_timed_and_sized(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with timed_load('tiny') as timing:
            model = load_tiny()

        assert timing[0].load_s > 0
        assert timing[0].model_id == 'tiny'
        assert count_parameters(model) > 0

    def test_stages_are_carried_into_the_row(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with timed_load('tiny') as timing:
            load_tiny()

        row = timing[0].with_stages({'load_model': 1.5}).to_row()
        assert row['load_load_model_s'] == 1.5
        assert 'stages' not in row

    def test_savings_are_reported_once_a_load_is_measured(self, harness, tmp_path):
        result = run(harness)
        result.load_s = 240.0
        savings = result.savings_report(32)

        assert savings['saved_s'] == pytest.approx(31 * 240.0)
        assert savings['speedup'] > 1

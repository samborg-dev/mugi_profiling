import math

import pytest
import torch

from profiling_api.evaluate import WindowEvalHarness, count_tokens, hash_batches
from profiling_api.windows import ATTENTION, FFN, FfnWindow, SoftmaxWindow, WindowAssignment
from tiny_host import TINY_LLAMA, TinyHost, harness, host, load_tiny  # noqa: F401


class TestSetup:
    def test_baseline_is_measured_before_patching(self, host):
        h = WindowEvalHarness(host)
        assert host.nonlinear_sites == []
        h.setup()
        assert h.baseline_ppl is not None and math.isfinite(h.baseline_ppl)
        assert len(host.nonlinear_sites) == host.model.config.num_hidden_layers

    def test_setup_twice_is_refused(self, harness):
        with pytest.raises(RuntimeError, match="already ran"):
            harness.setup()

    def test_training_mode_is_refused(self, host):
        host.model.train()
        with pytest.raises(RuntimeError, match="training mode"):
            WindowEvalHarness(host).setup()

    def test_non_eager_attention_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = load_tiny(attn_implementation='sdpa')
        model.eval()
        with pytest.raises(RuntimeError, match="attn_implementation"):
            WindowEvalHarness(TinyHost(model, tmp_path)).setup()

    def test_the_softmax_lut_is_actually_reached(self, harness):
        obj = harness.host.nonlinear_sites[0][ATTENTION]
        calls = {'n': 0}
        original = obj.nonlinear

        def spy(*a, **k):
            calls['n'] += 1
            return original(*a, **k)

        obj.nonlinear = spy
        harness.evaluate(harness.uniform())
        assert calls['n'] == len(harness.host.inputs)

    def test_profiling_is_disabled_by_default(self, harness):
        assert all(e[s].profiling_enabled is False
                   for e in harness.host.nonlinear_sites for s in (ATTENTION, FFN))

    def test_profiling_can_be_left_on(self, host):
        h = WindowEvalHarness(host, profiling='on')
        h.setup()
        assert all(e[s].profiling_enabled is True
                   for e in host.nonlinear_sites for s in (ATTENTION, FFN))


class TestEvaluate:
    def test_returns_a_finite_perplexity(self, harness):
        r = harness.evaluate(harness.uniform())
        assert math.isfinite(r.ppl) and r.ppl > 0

    def test_runs_many_times_without_reloading(self, harness):
        model_id = id(harness.host.model)
        sites_id = id(harness.host.nonlinear_sites)

        for anchor in range(-4, 4):
            harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=8, anchor=anchor)))

        assert id(harness.host.model) == model_id
        assert id(harness.host.nonlinear_sites) == sites_id
        assert len(harness.results) == 8

    def test_a_starved_window_differs_from_a_covering_one(self, harness):
        starved = harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=1, anchor=4)))
        covering = harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=8, anchor=0)))
        assert starved.ppl != covering.ppl
        assert starved.ppl > covering.ppl

    def test_widening_past_full_coverage_changes_nothing(self, harness):
        wide = harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=16, anchor=4)))
        wider = harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=16, anchor=2)))
        assert wide.ppl == wider.ppl

    def test_shrinking_the_ceiling_below_the_data_destroys_the_model(self, harness):
        r = harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=8, anchor=-4)))
        assert math.isnan(r.ppl)

    def test_patched_model_differs_from_the_unpatched_baseline(self, harness):
        assert harness.evaluate(harness.uniform()).ppl != harness.baseline_ppl

    def test_runs_a_full_anchor_sweep_in_one_process(self, harness):
        anchors = (4, 2, 0, -2)
        ppls = [harness.evaluate(harness.uniform(
            attention=SoftmaxWindow(exp_dim=8, anchor=a))).ppl for a in anchors]

        assert len(ppls) == len(anchors)
        finite = [p for p in ppls if math.isfinite(p)]
        assert finite, f"no anchor produced a finite perplexity: {ppls}"
        assert len(set(ppls)) > 1

    def test_same_assignment_is_deterministic_on_cpu(self, harness):
        a = harness.uniform(attention=SoftmaxWindow(exp_dim=8, anchor=0))
        first = harness.evaluate(a)
        second = harness.evaluate(a)

        assert first.assignment_hash == second.assignment_hash
        assert first.ppl == second.ppl

    def test_per_layer_assignment_differs_from_uniform(self, harness):
        n = harness.n_layers
        uniform = harness.uniform(attention=SoftmaxWindow(exp_dim=8, anchor=0))

        varied = WindowAssignment()
        for layer in range(n):
            varied.set(layer, ATTENTION, SoftmaxWindow(exp_dim=8, anchor=2 - layer))
            varied.set(layer, FFN, harness.seed_ffn)

        assert harness.evaluate(uniform).ppl != harness.evaluate(varied).ppl

    def test_evaluate_before_setup_is_refused(self, host):
        h = WindowEvalHarness(host)
        with pytest.raises(RuntimeError, match="call setup"):
            h.evaluate(WindowAssignment())

    def test_no_profile_files_are_written_during_search(self, harness, tmp_path):
        before = set(tmp_path.rglob('*.pt'))
        for anchor in (-2, 0, 2):
            harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=8, anchor=anchor)))
        assert set(tmp_path.rglob('*.pt')) == before


class TestResultMetadata:
    def test_input_hash_is_stable_across_evals(self, harness):
        a = harness.evaluate(harness.uniform())
        b = harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=4, anchor=1)))
        assert a.input_hash == b.input_hash == harness.input_hash

    def test_assignment_hash_tracks_the_windows(self, harness):
        a = harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=8, anchor=1)))
        b = harness.evaluate(harness.uniform(attention=SoftmaxWindow(exp_dim=8, anchor=-1)))
        assert a.assignment_hash != b.assignment_hash

    def test_cost_fields_are_populated(self, harness):
        r = harness.evaluate(harness.uniform())
        assert r.wall_s > 0
        assert r.apply_ms >= 0
        assert r.n_tokens == count_tokens(harness.host.inputs)
        assert r.num_batches == len(harness.host.inputs)

    def test_gpu_fields_are_none_on_cpu(self, harness):
        r = harness.evaluate(harness.uniform())
        assert r.gpu_s is None
        assert r.peak_mem_bytes == {}

    def test_provenance_records_the_stack(self, harness):
        r = harness.evaluate(harness.uniform())
        assert r.provenance['function_name'] == 'vlp'
        assert r.provenance['profiling'] == 'off'
        assert r.provenance['torch'] == torch.__version__

    def test_token_weighted_ppl_matches_when_batches_are_equal_length(self, harness):
        r = harness.evaluate(harness.uniform())
        assert r.ppl_token_weighted == pytest.approx(r.ppl)


class TestCsvOutput:
    def test_dataframe_includes_the_baseline_row(self, harness):
        harness.evaluate(harness.uniform())
        df = harness.to_dataframe()
        assert len(df) == 2
        assert df.iloc[0]['assignment_hash'] == 'baseline-unpatched'

    def test_write_csv_roundtrips(self, harness, tmp_path):
        import pandas as pd

        harness.evaluate(harness.uniform())
        path = harness.write_csv(str(tmp_path / 'out' / 'cost.csv'))
        assert len(pd.read_csv(path)) == 2


class TestBatchHashing:
    def test_hash_is_order_sensitive(self, host):
        first = hash_batches(host.inputs)
        host.inputs = list(reversed(host.inputs))
        assert hash_batches(host.inputs) != first

    def test_hash_is_stable_for_identical_batches(self, host):
        assert hash_batches(host.inputs) == hash_batches(host.inputs)

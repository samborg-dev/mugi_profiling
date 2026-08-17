import math

import pytest
import yaml

from run_window_search import bootstrap_ci, parse_exp_dims, resolve_noise_floor, summarise


class FakeEval:
    def __init__(self, ppl, wall_s=1.0, apply_ms=1.0, batch_losses=()):
        self.ppl = ppl
        self.wall_s = wall_s
        self.apply_ms = apply_ms
        self.batch_losses = tuple(batch_losses)
        self.n_tokens = 4096
        self.num_batches = len(self.batch_losses)


class Args:
    def __init__(self, **kwargs):
        self.noise_floor = None
        self.noise_summary = None
        self.noise_metric = 'ci_width'
        self.__dict__.update(kwargs)


def write_summary(tmp_path, **fields):
    path = tmp_path / 'summary.yaml'
    path.write_text(yaml.safe_dump(fields))
    return str(path)


class TestDeterministicEvalHasNoSpread:
    def test_repeating_an_identical_eval_gives_exactly_zero_spread(self):
        stats = summarise([FakeEval(6.28) for _ in range(10)])
        assert stats['ppl_spread'] == 0.0
        assert stats['ppl_stdev'] == 0.0

    def test_the_bootstrap_still_finds_noise_in_the_same_run(self):
        losses = [1.8 + 0.1 * math.sin(i) for i in range(64)]
        ci = bootstrap_ci(losses, n_resamples=2000)
        assert ci['ppl_ci_width'] > 0.0
        assert ci['ppl_ci_low'] < ci['ppl_ci_high']

    def test_the_interval_brackets_the_point_estimate(self):
        losses = [1.8 + 0.1 * math.sin(i) for i in range(64)]
        ci = bootstrap_ci(losses, n_resamples=2000)
        point = math.exp(sum(losses) / len(losses))
        assert ci['ppl_ci_low'] <= point <= ci['ppl_ci_high']

    def test_more_batches_narrow_the_interval(self):
        few = bootstrap_ci([1.8 + 0.1 * math.sin(i) for i in range(16)], n_resamples=2000)
        many = bootstrap_ci([1.8 + 0.1 * math.sin(i) for i in range(256)], n_resamples=2000)
        assert many['ppl_ci_width'] < few['ppl_ci_width']

    def test_a_host_that_records_no_batches_yields_nan_rather_than_zero(self):
        ci = bootstrap_ci(())
        assert math.isnan(ci['ppl_ci_width'])


class TestResolveNoiseFloor:
    def test_the_ci_width_is_the_default_source(self, tmp_path):
        path = write_summary(tmp_path, ppl_spread=0.0, ppl_ci_width=0.004)
        floor, source = resolve_noise_floor(Args(noise_summary=path))
        assert floor == pytest.approx(0.004)
        assert 'ci_width' in source

    def test_half_width_halves_the_floor(self, tmp_path):
        path = write_summary(tmp_path, ppl_ci_width=0.004)
        floor, _ = resolve_noise_floor(
            Args(noise_summary=path, noise_metric='ci_half_width'))
        assert floor == pytest.approx(0.002)

    def test_an_explicit_floor_wins(self, tmp_path):
        path = write_summary(tmp_path, ppl_ci_width=0.004)
        floor, source = resolve_noise_floor(Args(noise_summary=path, noise_floor=0.01))
        assert floor == 0.01
        assert '--noise_floor' in source

    def test_a_structurally_zero_floor_is_refused(self, tmp_path):
        path = write_summary(tmp_path, ppl_spread=0.0, ppl_ci_width=0.004)
        with pytest.raises(SystemExit, match='accept any improvement'):
            resolve_noise_floor(Args(noise_summary=path, noise_metric='spread'))

    def test_the_refusal_explains_why_spread_is_zero(self, tmp_path):
        path = write_summary(tmp_path, ppl_spread=0.0)
        with pytest.raises(SystemExit, match='deterministic'):
            resolve_noise_floor(Args(noise_summary=path, noise_metric='spread'))

    def test_a_nan_floor_is_refused(self, tmp_path):
        path = write_summary(tmp_path, ppl_ci_width=float('nan'))
        with pytest.raises(SystemExit, match='batch_losses'):
            resolve_noise_floor(Args(noise_summary=path))

    def test_a_summary_missing_the_metric_is_rejected(self, tmp_path):
        path = write_summary(tmp_path, ppl_spread=0.01)
        with pytest.raises(ValueError, match='ppl_ci_width'):
            resolve_noise_floor(Args(noise_summary=path))

    def test_no_floor_and_no_summary_is_fatal(self):
        with pytest.raises(SystemExit, match='needs a noise floor'):
            resolve_noise_floor(Args())


class TestParseExpDims:
    def test_a_comma_list_becomes_a_tuple(self):
        assert parse_exp_dims('8,10,12') == (8, 10, 12)

    def test_whitespace_is_tolerated(self):
        assert parse_exp_dims(' 8 , 10 ') == (8, 10)

    def test_an_empty_spec_means_no_sweep(self):
        assert parse_exp_dims(None) is None
        assert parse_exp_dims('') is None

    def test_a_non_positive_window_size_is_rejected(self):
        with pytest.raises(ValueError, match='exp_dim'):
            parse_exp_dims('10,0')

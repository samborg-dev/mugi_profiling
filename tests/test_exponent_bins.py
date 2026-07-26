import os

import pytest
import torch

from exponent_bins import (
    EXP_BIAS,
    LEGACY_EXP_BIAS,
    LEGACY_N_BINS,
    N_BINS,
    WINDOW_BINS,
    analysis_window,
    exp_to_index,
    index_to_exp,
    normalize_window,
)


def write_histogram(exponent_counts):
    vals = []
    for e, count in exponent_counts.items():
        vals += [torch.ldexp(torch.tensor(0.75), torch.tensor(e))] * count
    values = torch.stack(vals).to(torch.bfloat16)

    mant, exp = torch.frexp(values)
    del mant
    exp = torch.where(values != 0, exp + EXP_BIAS, torch.zeros_like(exp)).flatten()
    return torch.bincount(exp, minlength=N_BINS)


@pytest.fixture
def profile_leaf(tmp_path):
    def _make(hist, nonlinear="softmax", layer=0, seq_len=2):
        leaf = tmp_path / nonlinear / f"pre_{nonlinear}" / "exp_dist" / f"layer_{layer}"
        leaf.mkdir(parents=True, exist_ok=True)
        path = leaf / f"seq_len_{seq_len}.pt"
        torch.save(hist, path)
        return str(path)
    return _make


class TestBinConvention:
    def test_writer_puts_exponent_e_in_bin_bias_plus_e(self):
        hist = write_histogram({-3: 5, 0: 7, 4: 2})
        assert hist[EXP_BIAS - 3] == 5
        assert hist[EXP_BIAS + 0] == 7
        assert hist[EXP_BIAS + 4] == 2
        assert hist.sum() == 14

    def test_analysis_window_is_30_bins_and_recovers_planted_exponents(self):
        planted = {-5: 3, -1: 11, 2: 6}
        window = analysis_window(write_histogram(planted))

        assert len(window) == WINDOW_BINS == 30
        recovered = {index_to_exp(int(i)): int(window[i]) for i in (window > 0).nonzero().flatten()}
        assert recovered == planted

    def test_index_and_exp_conversions_are_inverses(self):
        for i in range(WINDOW_BINS):
            assert exp_to_index(index_to_exp(i)) == i

    def test_the_old_slice_would_have_been_empty(self):
        hist = write_histogram({-4: 10, 0: 20, 3: 5})
        assert hist[1:31].sum() == 0
        assert analysis_window(hist).sum() == 35

    def test_exponent_zero_is_not_conflated_with_exact_zeros(self):
        values = torch.tensor([0.0, 0.0, 0.0, 0.75, 0.5, 0.9], dtype=torch.bfloat16)
        mant, exp = torch.frexp(values)
        exp = torch.where(values != 0, exp + EXP_BIAS, torch.zeros_like(exp)).flatten()
        hist = torch.bincount(exp, minlength=N_BINS)

        assert hist[0] == 3
        assert hist[EXP_BIAS + 0] == 3
        assert analysis_window(hist)[exp_to_index(0)] == 3


class TestLegacyLayout:
    def test_legacy_32_bin_dumps_still_analyse(self):
        hist = torch.zeros(LEGACY_N_BINS)
        hist[LEGACY_EXP_BIAS - 2] = 40.0
        window = analysis_window(hist)
        assert len(window) == WINDOW_BINS
        assert window[exp_to_index(-2)] == 40.0

    def test_unrecognised_length_is_rejected(self):
        with pytest.raises(ValueError, match="unrecognised exponent histogram length"):
            analysis_window(torch.zeros(64))

    def test_multidimensional_input_is_rejected(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            analysis_window(torch.zeros(2, N_BINS))


class TestEmptyWindowIsAnError:
    def test_all_zero_window_raises_instead_of_returning_nan(self):
        with pytest.raises(ValueError, match="empty across the analysis window"):
            normalize_window(torch.zeros(WINDOW_BINS))

    def test_normalize_sums_to_100(self):
        window = analysis_window(write_histogram({-2: 3, 1: 1}))
        assert normalize_window(window).sum().item() == pytest.approx(100.0)


class TestReadersRecoverPlantedPeak:
    def test_profile_tensor_runs_and_recovers_argmax(self, profile_leaf):
        from profile_distribution import profile_tensor

        path = profile_leaf(write_histogram({-4: 2, -3: 8, -2: 30, -1: 9, 0: 3}))
        out = profile_tensor(path)

        assert out["argmax_value"] == -2
        assert out["max_exp"] == 10 and out["min_exp"] == 10
        assert out["cluster"] == "max_cluster"

    def test_window_sizer_generic_recovers_argmax_and_brackets_it(self, profile_leaf):
        from profiling_api.config import ProfileConfig
        from profiling_api.window import WindowSizer

        path = profile_leaf(write_histogram({-6: 2, -4: 25, -3: 60, -2: 90, -1: 70, 0: 30, 2: 3}))
        spec = WindowSizer(ProfileConfig(model_id="x"))._size_tensor(path)

        assert spec["argmax_value"] == -2
        assert spec["min_exp"] <= spec["argmax_value"] <= spec["max_exp"]

    def test_profile_tensor_accepts_os_path_join_paths(self, profile_leaf):
        from profile_distribution import profile_tensor

        path = profile_leaf(write_histogram({-2: 10}), nonlinear="softmax", layer=7)
        assert os.sep in path or "/" in path
        out = profile_tensor(os.path.normpath(path))
        assert out["cluster"] in ("max_cluster", "min_cluster")


class TestCalibrationScopeGuard:
    def test_layer_beyond_31_raises_a_clear_error(self, profile_leaf):
        from profile_distribution import profile_tensor

        path = profile_leaf(write_histogram({-2: 10}), layer=39)
        with pytest.raises(ValueError, match="no entry for layer=39"):
            profile_tensor(path)

    def test_unknown_nonlinear_raises_a_clear_error(self, profile_leaf):
        from profile_distribution import profile_tensor

        path = profile_leaf(write_histogram({-2: 10}), nonlinear="mish")
        with pytest.raises(ValueError, match="no entry for layer=0"):
            profile_tensor(path)

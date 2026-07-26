import pytest
import torch

from custom_nonlinear.custom_nonlinear_functions.vlp_gelu_approx import VLPGelu
from custom_nonlinear.custom_nonlinear_functions.vlp_silu_approx import VLPSilu
from custom_nonlinear.custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax
from profiling_api.windows import FfnWindow, SoftmaxWindow

COMMON = dict(layer=0, device='cpu', profile_dims=[0], keys=[])


def make_softmax(window, tmp_path):
    return VLPSoftmax(**window.to_kwargs(), profile_path=str(tmp_path) + '/', **COMMON)


def make_ffn(cls, window, tmp_path):
    return cls(**window.to_kwargs(), profile_path=str(tmp_path) + '/', **COMMON)


class TestSoftmaxResetEqualsFreshConstruction:
    @pytest.mark.parametrize("exp_dim,anchor,side", [
        (8, 2, "max"), (16, 1, "max"), (8, -5, "min"), (16, -12, "min"),
        (4, 0, "max"), (1, 3, "max"),
    ])
    def test_lut_is_bit_identical(self, exp_dim, anchor, side, tmp_path):
        start = SoftmaxWindow(exp_dim=16, anchor=-7, anchor_side="min")
        target = SoftmaxWindow(exp_dim=exp_dim, anchor=anchor, anchor_side=side)

        fresh = make_softmax(target, tmp_path)
        reset = make_softmax(start, tmp_path)
        reset.reset_lut(**target.to_kwargs())

        assert torch.equal(reset.lut, fresh.lut)
        assert reset.lut.dtype == fresh.lut.dtype
        assert reset.lut.shape == fresh.lut.shape

    def test_derived_end_matches_the_schema(self, tmp_path):
        w = SoftmaxWindow(exp_dim=8, anchor=2, anchor_side="max")
        obj = make_softmax(w, tmp_path)
        assert obj.max_exp == w.max_exp
        assert obj.min_exp == w.min_exp

    def test_min_anchored_derived_end_matches_the_schema(self, tmp_path):
        w = SoftmaxWindow(exp_dim=8, anchor=-5, anchor_side="min")
        obj = make_softmax(w, tmp_path)
        assert obj.min_exp == w.min_exp
        assert obj.max_exp == w.max_exp

    def test_repeated_reset_is_idempotent(self, tmp_path):
        w = SoftmaxWindow(exp_dim=8, anchor=1, anchor_side="max")
        obj = make_softmax(w, tmp_path)
        first = obj.lut.clone()
        for _ in range(5):
            obj.reset_lut(**w.to_kwargs())
        assert torch.equal(obj.lut, first)

    def test_reset_round_trip_returns_to_the_original(self, tmp_path):
        a = SoftmaxWindow(exp_dim=8, anchor=1, anchor_side="max")
        b = SoftmaxWindow(exp_dim=16, anchor=-4, anchor_side="min")

        obj = make_softmax(a, tmp_path)
        original = obj.lut.clone()
        obj.reset_lut(**b.to_kwargs())
        obj.reset_lut(**a.to_kwargs())

        assert torch.equal(obj.lut, original)

    def test_different_windows_give_different_luts(self, tmp_path):
        a = make_softmax(SoftmaxWindow(exp_dim=8, anchor=1), tmp_path)
        b = make_softmax(SoftmaxWindow(exp_dim=8, anchor=-4), tmp_path)
        assert not torch.equal(a.lut, b.lut)


@pytest.mark.parametrize("cls", [VLPSilu, VLPGelu])
class TestFfnResetEqualsFreshConstruction:
    @pytest.mark.parametrize("exp_dim,pos,neg", [
        (8, 2, 2), (16, 1, -3), (8, -2, 4), (4, 0, 0),
    ])
    def test_both_luts_are_bit_identical(self, cls, exp_dim, pos, neg, tmp_path):
        start = FfnWindow(exp_dim=16, pos_anchor=-7, neg_anchor=5)
        target = FfnWindow(exp_dim=exp_dim, pos_anchor=pos, neg_anchor=neg)

        fresh = make_ffn(cls, target, tmp_path)
        reset = make_ffn(cls, start, tmp_path)
        reset.reset_lut(**target.to_kwargs())

        assert torch.equal(reset.pos_lut, fresh.pos_lut)
        assert torch.equal(reset.neg_lut, fresh.neg_lut)

    def test_derived_floors_match_the_schema(self, cls, tmp_path):
        w = FfnWindow(exp_dim=8, pos_anchor=2, neg_anchor=-1)
        obj = make_ffn(cls, w, tmp_path)
        assert obj.pos_min_exp == w.pos_min_exp
        assert obj.neg_min_exp == w.neg_min_exp

    def test_kwargs_are_accepted_by_the_constructor(self, cls, tmp_path):
        make_ffn(cls, FfnWindow(exp_dim=16, pos_anchor=2, neg_anchor=2), tmp_path)

    def test_lut_build_is_rejected(self, cls, tmp_path):
        kwargs = FfnWindow(exp_dim=8, pos_anchor=0, neg_anchor=0).to_kwargs()
        kwargs['lut_build'] = 'max'
        with pytest.raises(TypeError):
            cls(**kwargs, profile_path=str(tmp_path) + '/', **COMMON)


class TestProfilingFlag:
    def test_defaults_to_enabled(self, tmp_path):
        obj = make_softmax(SoftmaxWindow(exp_dim=8, anchor=0), tmp_path)
        assert obj.profiling_enabled is True

    def test_disabled_profile_writes_nothing(self, tmp_path):
        obj = make_softmax(SoftmaxWindow(exp_dim=8, anchor=0), tmp_path)
        obj.profiling_enabled = False

        before = set(p for p in tmp_path.rglob('*.pt'))
        obj.profile(torch.randn(4, 8), dim=-1, profile_path='x',
                    left_value_edge=-20.25, right_value_edge=20, value_index=-0.05)
        assert set(p for p in tmp_path.rglob('*.pt')) == before

    def test_enabled_profile_writes_files(self, tmp_path):
        obj = make_softmax(SoftmaxWindow(exp_dim=8, anchor=0), tmp_path)
        obj.profile(torch.randn(4, 8), dim=-1, profile_path='x',
                    left_value_edge=-20.25, right_value_edge=20, value_index=-0.05)
        assert list(tmp_path.rglob('*.pt'))

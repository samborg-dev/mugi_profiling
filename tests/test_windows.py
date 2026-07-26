import pytest

from profiling_api.windows import (
    ATTENTION,
    FFN,
    FfnWindow,
    SoftmaxWindow,
    WindowAssignment,
    window_for_site,
)


class TestSoftmaxWindow:
    def test_max_anchored_derives_the_floor(self):
        w = SoftmaxWindow(exp_dim=8, anchor=2, anchor_side="max")
        assert w.max_exp == 2
        assert w.min_exp == -5
        assert list(w.covers) == list(range(-5, 3))

    def test_min_anchored_derives_the_ceiling(self):
        w = SoftmaxWindow(exp_dim=8, anchor=-5, anchor_side="min")
        assert w.min_exp == -5
        assert w.max_exp == 2

    def test_exp_dim_one_is_a_single_exponent(self):
        w = SoftmaxWindow(exp_dim=1, anchor=3, anchor_side="max")
        assert w.max_exp == w.min_exp == 3

    def test_kwargs_match_the_constructor(self):
        w = SoftmaxWindow(exp_dim=16, anchor=1, anchor_side="max", group_size=32)
        assert w.to_kwargs() == dict(exp_dim=16, max_exp=1, min_exp=-14,
                                     window_size=32, lut_build="max")

    def test_from_profile_uses_the_cluster_verdict(self):
        assert SoftmaxWindow.from_profile(max_exp=3, min_exp=-5,
                                          cluster="max_cluster").anchor_side == "max"
        assert SoftmaxWindow.from_profile(max_exp=3, min_exp=-5,
                                          cluster="min_cluster").anchor_side == "min"

    def test_from_profile_anchors_on_the_live_end(self):
        assert SoftmaxWindow.from_profile(3, -5, "max_cluster").anchor == 3
        assert SoftmaxWindow.from_profile(3, -5, "min_cluster").anchor == -5

    @pytest.mark.parametrize("bad", [dict(exp_dim=0, anchor=1),
                                     dict(exp_dim=-1, anchor=1),
                                     dict(exp_dim=8, anchor=1, anchor_side="middle"),
                                     dict(exp_dim=8, anchor=1, group_size=0)])
    def test_invalid_fields_are_rejected(self, bad):
        with pytest.raises(ValueError):
            SoftmaxWindow(**bad)

    def test_is_immutable(self):
        w = SoftmaxWindow(exp_dim=8, anchor=0)
        with pytest.raises(Exception):
            w.anchor = 5


class TestFfnWindow:
    def test_both_sides_are_max_anchored(self):
        w = FfnWindow(exp_dim=8, pos_anchor=2, neg_anchor=-1)
        assert w.pos_min_exp == -5
        assert w.neg_min_exp == -8

    def test_kwargs_match_the_constructor(self):
        w = FfnWindow(exp_dim=16, pos_anchor=2, neg_anchor=2, group_size=32)
        assert w.to_kwargs() == dict(exp_dim=16, max_pos_exp=2, max_neg_exp=2,
                                     window_size=32)

    def test_no_lut_build_key_is_emitted(self):
        assert "lut_build" not in FfnWindow(exp_dim=8, pos_anchor=0, neg_anchor=0).to_kwargs()

    def test_from_profile_gives_both_sides_the_same_anchor(self):
        w = FfnWindow.from_profile(max_exp=2)
        assert w.pos_anchor == w.neg_anchor == 2


class TestWindowForSite:
    def test_dispatches_on_site(self):
        assert isinstance(window_for_site(ATTENTION, exp_dim=8, anchor=0), SoftmaxWindow)
        assert isinstance(window_for_site(FFN, exp_dim=8, pos_anchor=0, neg_anchor=0), FfnWindow)

    def test_unknown_site_is_rejected(self):
        with pytest.raises(ValueError, match="unknown site"):
            window_for_site("mlp", exp_dim=8, anchor=0)


class TestWindowAssignment:
    @staticmethod
    def _uniform(n=4):
        return WindowAssignment.uniform(
            n,
            attention=SoftmaxWindow(exp_dim=16, anchor=1),
            ffn=FfnWindow(exp_dim=16, pos_anchor=2, neg_anchor=2),
        )

    def test_uniform_covers_every_layer_and_site(self):
        a = self._uniform(4)
        assert len(a) == 8
        assert a.layers == [0, 1, 2, 3]

    def test_site_type_is_enforced(self):
        a = WindowAssignment()
        with pytest.raises(TypeError, match="needs a SoftmaxWindow"):
            a.set(0, ATTENTION, FfnWindow(exp_dim=8, pos_anchor=0, neg_anchor=0))
        with pytest.raises(TypeError, match="needs a FfnWindow"):
            a.set(0, FFN, SoftmaxWindow(exp_dim=8, anchor=0))

    def test_with_override_does_not_mutate_the_original(self):
        a = self._uniform(4)
        b = a.with_override(2, ATTENTION, SoftmaxWindow(exp_dim=8, anchor=-3))

        assert a.get(2, ATTENTION).anchor == 1
        assert b.get(2, ATTENTION).anchor == -3
        assert b.get(0, ATTENTION).anchor == 1

    def test_with_field_changes_one_knob(self):
        a = self._uniform(2)
        b = a.with_field(1, ATTENTION, anchor=-4)

        assert b.get(1, ATTENTION).anchor == -4
        assert b.get(1, ATTENTION).exp_dim == 16
        assert a.get(1, ATTENTION).anchor == 1

    def test_items_are_ordered_deterministically(self):
        a = self._uniform(3)
        keys = [(layer, site) for layer, site, _ in a.items()]
        assert keys == sorted(keys)

    def test_digest_is_stable_across_construction_order(self):
        forward = WindowAssignment()
        backward = WindowAssignment()
        for layer in range(3):
            forward.set(layer, ATTENTION, SoftmaxWindow(exp_dim=16, anchor=layer))
        for layer in reversed(range(3)):
            backward.set(layer, ATTENTION, SoftmaxWindow(exp_dim=16, anchor=layer))

        assert forward.digest() == backward.digest()

    def test_digest_changes_when_any_window_changes(self):
        a = self._uniform(4)
        assert a.digest() != a.with_field(2, ATTENTION, anchor=-3).digest()

    def test_digest_ignores_the_derived_end(self):
        same = SoftmaxWindow(exp_dim=8, anchor=2, anchor_side="max")
        a = WindowAssignment({(0, ATTENTION): same})
        b = WindowAssignment({(0, ATTENTION): SoftmaxWindow(exp_dim=8, anchor=2)})
        assert a.digest() == b.digest()

    def test_roundtrips_through_dict(self):
        a = self._uniform(4)
        assert WindowAssignment.from_dict(a.to_dict()) == a

    def test_roundtrips_through_yaml(self, tmp_path):
        a = self._uniform(4)
        path = a.to_yaml(str(tmp_path / "assignment.yaml"))
        loaded = WindowAssignment.from_yaml(path)

        assert loaded == a
        assert loaded.digest() == a.digest()

import os

import pytest
import yaml

from custom_nonlinear.custom_nonlinear_functions.vlp_gelu_approx import VLPGelu
from custom_nonlinear.custom_nonlinear_functions.vlp_silu_approx import VLPSilu
from custom_nonlinear.custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax
from profile_distribution import attention_params, create_nonlinear_config, ffn_params

PROFILE = {
    'argmax_value': -2,
    'tensor': [1.0, 2.0],
    'max_exp': 2,
    'min_exp': -5,
    'max_exp_unclamped': 3,
    'min_exp_unclamped': -6,
    'mean': 1.5,
    'median': 1.5,
    'centroid': 0.5,
    'cluster': 'max_cluster',
}

COMMON = dict(layer=0, device='cpu', profile_dims=[0], keys=[])


@pytest.fixture
def distribution_tree(tmp_path):
    def _make(nonlinears=('softmax', 'silu'), layers=2, profile=None):
        for nl in nonlinears:
            for layer in range(layers):
                leaf = tmp_path / nl / f"layer_{layer}"
                leaf.mkdir(parents=True, exist_ok=True)
                with open(leaf / 'profile.yaml', 'w') as f:
                    yaml.safe_dump(profile or PROFILE, f)
        return str(tmp_path)
    return _make


class TestParamHelpers:
    def test_attention_keys_match_the_constructor(self, tmp_path):
        VLPSoftmax(**attention_params(PROFILE), profile_path=str(tmp_path) + '/', **COMMON)

    @pytest.mark.parametrize("cls", [VLPSilu, VLPGelu])
    def test_ffn_keys_match_the_constructor(self, cls, tmp_path):
        cls(**ffn_params(PROFILE), profile_path=str(tmp_path) + '/', **COMMON)

    def test_ffn_no_longer_emits_min_pos_exp(self):
        assert 'min_pos_exp' not in ffn_params(PROFILE)

    def test_ffn_no_longer_emits_lut_build(self):
        assert 'lut_build' not in ffn_params(PROFILE)

    def test_ffn_anchors_both_sides_at_the_profile_max(self):
        p = ffn_params(PROFILE)
        assert p['max_pos_exp'] == p['max_neg_exp'] == PROFILE['max_exp']

    def test_attention_lut_build_follows_the_cluster(self):
        assert attention_params({**PROFILE, 'cluster': 'max_cluster'})['lut_build'] == 'max'
        assert attention_params({**PROFILE, 'cluster': 'min_cluster'})['lut_build'] == 'min'

    def test_exp_dim_and_window_size_are_overridable(self):
        p = attention_params(PROFILE, exp_dim=8, window_size=256)
        assert p['exp_dim'] == 8 and p['window_size'] == 256


class TestCreateNonlinearConfig:
    def test_emits_one_entry_per_layer(self, distribution_tree):
        cfg = create_nonlinear_config(distribution_tree(layers=3), 'llama')
        assert sorted(cfg['params']) == ['0', '1', '2']

    def test_every_emitted_block_constructs(self, distribution_tree, tmp_path):
        cfg = create_nonlinear_config(distribution_tree(layers=2), 'llama')
        out = tmp_path / 'built'
        out.mkdir()

        for layer, params in cfg['params'].items():
            VLPSoftmax(**params['vlp']['attention'],
                       profile_path=str(out) + '/', **COMMON)
            VLPSilu(**params['vlp']['ffn'],
                    profile_path=str(out) + '/', **COMMON)

    def test_no_stale_ffn_keys_survive_anywhere(self, distribution_tree):
        cfg = create_nonlinear_config(distribution_tree(layers=3), 'llama')
        for params in cfg['params'].values():
            assert set(params['vlp']['ffn']) == {
                'exp_dim', 'max_pos_exp', 'max_neg_exp', 'window_size'}

    def test_attention_block_keeps_lut_build(self, distribution_tree):
        cfg = create_nonlinear_config(distribution_tree(layers=1), 'llama')
        assert cfg['params']['0']['vlp']['attention']['lut_build'] == 'max'

    def test_min_anchored_layers_round_trip(self, distribution_tree, tmp_path):
        tree = distribution_tree(layers=1, profile={**PROFILE, 'cluster': 'min_cluster'})
        cfg = create_nonlinear_config(tree, 'llama')
        attn = cfg['params']['0']['vlp']['attention']

        assert attn['lut_build'] == 'min'
        obj = VLPSoftmax(**attn, profile_path=str(tmp_path) + '/', **COMMON)
        assert obj.min_exp == PROFILE['min_exp']

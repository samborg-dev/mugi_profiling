import os

import pytest
import torch

from exponent_bins import EXP_BIAS, LEGACY_EXP_BIAS, LEGACY_N_BINS, N_BINS
from profiling_api.seed import (FALLBACK_ANCHOR, SEED_SOURCE_FALLBACK,
                                SEED_SOURCE_HISTOGRAM, HistogramSeeder,
                                fallback_assignment)
from profiling_api.windows import ATTENTION, FFN, SoftmaxWindow

MASS = {0: 50, -1: 30, -2: 12, -3: 5, -4: 3}


def histogram(mass, n_bins=N_BINS, bias=EXP_BIAS):
    hist = torch.zeros(n_bins, dtype=torch.long)
    for exp, count in mass.items():
        hist[bias + exp] = count
    return hist


def write_layer(root, layer, mass, seq_len=63, name=None, **kwargs):
    layer_dir = os.path.join(root, f"layer_{layer}")
    os.makedirs(layer_dir, exist_ok=True)
    path = os.path.join(layer_dir, name or f"seq_len_{seq_len}.pt")
    torch.save(histogram(mass, **kwargs), path)
    return path


@pytest.fixture
def profile_root(tmp_path):
    root = tmp_path / "profile" / "llama" / "softmax" / "pre_softmax" / "exp_dist"
    root.mkdir(parents=True)
    return str(tmp_path / "profile"), str(root)


class TestPlacement:
    def test_a_seeded_window_covers_the_profiled_mass(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, MASS)

        seed = HistogramSeeder(base).seed_layer(0)
        assert seed.source == SEED_SOURCE_HISTOGRAM
        assert seed.max_exp >= max(MASS) - 1
        assert 0 in seed.window.covers

    def test_the_seed_records_where_it_came_from(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, MASS, seq_len=127)

        seed = HistogramSeeder(base).seed_layer(0)
        assert "seq_len=127" in seed.detail
        assert seed.cluster in ("max_cluster", "min_cluster")
        assert seed.from_histogram

    def test_the_longest_sequence_wins(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, {4: 100}, seq_len=63)
        write_layer(root, 0, MASS, seq_len=255)

        seed = HistogramSeeder(base).seed_layer(0)
        assert "seq_len=255" in seed.detail

    def test_histograms_at_the_same_length_are_summed(self, profile_root):
        base, root = profile_root
        layer_dir = os.path.join(root, "layer_0", "block_0")
        os.makedirs(layer_dir)
        torch.save(histogram(MASS), os.path.join(layer_dir, "seq_len_63.pt"))
        other = os.path.join(root, "layer_0", "block_1")
        os.makedirs(other)
        torch.save(histogram(MASS), os.path.join(other, "seq_len_63.pt"))

        seed = HistogramSeeder(base).seed_layer(0)
        assert seed.source == SEED_SOURCE_HISTOGRAM
        assert "files=2" in seed.detail

    def test_the_window_geometry_follows_the_seeder(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, MASS)

        seed = HistogramSeeder(base, exp_dim=10, group_size=128).seed_layer(0)
        assert seed.window.exp_dim == 10
        assert seed.window.group_size == 128

    def test_legacy_histograms_are_accepted(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, {0: 50, -1: 20, -2: 10, -3: 5},
                    n_bins=LEGACY_N_BINS, bias=LEGACY_EXP_BIAS)

        assert HistogramSeeder(base).seed_layer(0).source == SEED_SOURCE_HISTOGRAM

    def test_an_anchor_outside_the_searchable_range_is_clamped_and_flagged(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, {12: 1000})

        seed = HistogramSeeder(base).seed_layer(0)
        lo, hi = HistogramSeeder(base).anchor_bounds
        assert lo <= seed.window.anchor <= hi
        assert "clamped" in seed.detail


class TestFallback:
    def test_a_missing_profile_root_falls_back(self, tmp_path):
        seeder = HistogramSeeder(str(tmp_path / "nothing"))
        seed = seeder.seed_layer(0)

        assert seed.source == SEED_SOURCE_FALLBACK
        assert seed.window.anchor == FALLBACK_ANCHOR
        assert "no exp_dist directory" in seed.detail

    def test_a_missing_layer_falls_back(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, MASS)

        seed = HistogramSeeder(base).seed_layer(7)
        assert seed.source == SEED_SOURCE_FALLBACK
        assert "layer_7" in seed.detail

    def test_an_empty_histogram_falls_back_instead_of_raising(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, {})

        seed = HistogramSeeder(base).seed_layer(0)
        assert seed.source == SEED_SOURCE_FALLBACK
        assert "empty" in seed.detail

    def test_a_malformed_histogram_falls_back(self, profile_root):
        base, root = profile_root
        layer_dir = os.path.join(root, "layer_0")
        os.makedirs(layer_dir)
        torch.save(torch.ones(100), os.path.join(layer_dir, "seq_len_63.pt"))

        seed = HistogramSeeder(base).seed_layer(0)
        assert seed.source == SEED_SOURCE_FALLBACK
        assert "unrecognised exponent histogram length" in seed.detail

    def test_a_layer_directory_with_no_tensors_falls_back(self, profile_root):
        base, root = profile_root
        os.makedirs(os.path.join(root, "layer_0"))

        seed = HistogramSeeder(base).seed_layer(0)
        assert seed.source == SEED_SOURCE_FALLBACK
        assert "no seq_len_*.pt" in seed.detail

    def test_the_fallback_window_is_configurable(self, tmp_path):
        custom = SoftmaxWindow(exp_dim=8, anchor=-3, anchor_side='min')
        seeder = HistogramSeeder(str(tmp_path), fallback=custom)
        assert seeder.seed_layer(0).window == custom


class TestAssignment:
    def test_every_layer_and_site_is_covered(self, profile_root):
        base, root = profile_root
        write_layer(root, 1, MASS)

        assignment, seeds = HistogramSeeder(base).seed_assignment(4)

        assert len(seeds) == 4
        for layer in range(4):
            assert assignment.get(layer, ATTENTION) is not None
            assert assignment.get(layer, FFN) is not None

    def test_partial_profiles_mix_sources(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, MASS)
        write_layer(root, 2, MASS)

        _, seeds = HistogramSeeder(base).seed_assignment(4)
        sources = [s.source for s in seeds]

        assert sources == [SEED_SOURCE_HISTOGRAM, SEED_SOURCE_FALLBACK,
                           SEED_SOURCE_HISTOGRAM, SEED_SOURCE_FALLBACK]

    def test_available_layers_lists_what_was_profiled(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, MASS)
        write_layer(root, 5, MASS)

        assert HistogramSeeder(base).available_layers() == [0, 5]

    def test_available_layers_is_empty_without_a_profile(self, tmp_path):
        assert HistogramSeeder(str(tmp_path)).available_layers() == []

    def test_seed_rows_are_flat_enough_for_yaml(self, profile_root):
        base, root = profile_root
        write_layer(root, 0, MASS)

        _, seeds = HistogramSeeder(base).seed_assignment(2)
        row = seeds[0].to_row()

        assert row['layer'] == 0 and row['site'] == ATTENTION
        assert set(row) >= {'anchor', 'anchor_side', 'source', 'detail'}
        assert all(isinstance(v, (int, float, str, type(None))) for v in row.values())


class TestSeedingDisabled:
    def test_the_ablation_puts_every_layer_on_the_same_window(self):
        window = SoftmaxWindow(exp_dim=16, anchor=2, anchor_side='max')
        assignment, seeds = fallback_assignment(4, attention=window)

        assert all(s.source == SEED_SOURCE_FALLBACK for s in seeds)
        assert all(assignment.get(l, ATTENTION) == window for l in range(4))
        assert all(s.detail == "seeding disabled" for s in seeds)

    def test_the_ablation_still_fills_the_ffn_site(self):
        assignment, _ = fallback_assignment(3)
        assert all(assignment.get(l, FFN) is not None for l in range(3))

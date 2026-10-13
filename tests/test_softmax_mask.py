import pytest
import torch

pytest.importorskip("transformers")

from custom_nonlinear.custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax
from profiling_api.windows import SoftmaxWindow

COMMON = dict(layer=0, device='cpu', profile_dims=[0], keys=[])
NEG_MASK = torch.finfo(torch.float16).min


def make_softmax(window, tmp_path):
    return VLPSoftmax(**window.to_kwargs(), profile_path=str(tmp_path) + '/', **COMMON)


def causal_scores(q, scale, seed=0):
    torch.manual_seed(seed)
    scores = (torch.randn(1, 2, q, q) * scale).to(torch.float16)
    mask = torch.triu(torch.full((q, q), NEG_MASK, dtype=torch.float16), 1)
    return scores + mask


WINDOWS = [
    SoftmaxWindow(exp_dim=10, anchor=2, anchor_side="max"),
    SoftmaxWindow(exp_dim=8, anchor=-3, anchor_side="min"),
    SoftmaxWindow(exp_dim=16, anchor=7, anchor_side="max"),
]


class TestMaskedEntriesGetNoWeight:
    @pytest.mark.parametrize("window", WINDOWS)
    def test_neg_inf_entries_are_exactly_zero(self, window, tmp_path):
        row = torch.tensor([0.0, -3.0, -6.0, NEG_MASK, float('-inf'), float('-inf')], dtype=torch.float16)
        x = row.expand(256, 6).contiguous().view(1, 1, 256, 6)
        out = make_softmax(window, tmp_path).nonlinear(x)[0, 0, 0].float()
        assert torch.all(out[3:] == 0)
        assert out[0] > 0.9

    @pytest.mark.parametrize("window", WINDOWS)
    def test_causal_mask_with_overflow_to_neg_inf(self, window, tmp_path):
        x = causal_scores(256, scale=12.0)
        assert torch.isneginf(x).any()
        out = make_softmax(window, tmp_path).nonlinear(x).float()
        future = torch.triu(torch.ones(256, 256, dtype=torch.bool), 1).expand_as(out)
        assert torch.all(out[future] == 0)
        torch.testing.assert_close(out.sum(-1), torch.ones(out.shape[:-1]), atol=2e-2, rtol=0)

    def test_finite_and_neg_inf_masks_give_the_same_output(self, tmp_path):
        x = causal_scores(256, scale=12.0)
        finite = torch.where(torch.isneginf(x), torch.tensor(NEG_MASK, dtype=torch.float16), x)
        k = make_softmax(WINDOWS[0], tmp_path)
        assert torch.equal(k.nonlinear(x), k.nonlinear(finite))

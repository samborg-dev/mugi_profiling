import pytest
import torch

transformers = pytest.importorskip("transformers")

from profiling_api.apply import apply_assignment, set_profiling
from profiling_api.windows import ATTENTION, FFN, FfnWindow, SoftmaxWindow, WindowAssignment

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class Host:
    def __init__(self, model, tmp_path):
        self.model = model
        self.model_name = 'llama'
        self.device = 'cpu'
        self.profiling_dims = [0]
        self.nonlinear_sites = []
        self._tmp = str(tmp_path) + '/'

    nonlinear_sites_by_layer = None


from inference_classes.inference_class import InferenceModel

Host.nonlinear_sites_by_layer = InferenceModel.nonlinear_sites_by_layer


@pytest.fixture
def patched(tmp_path):
    from custom_nonlinear.custom_nonlinear_functions.vlp_silu_approx import VLPSilu
    from custom_nonlinear.custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax
    from inference_classes.model_adapters import get_adapter

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
    except Exception as e:
        pytest.skip(f"could not load {TINY_LLAMA}: {type(e).__name__}")

    host = Host(model, tmp_path)
    seed_attn = SoftmaxWindow(exp_dim=16, anchor=1).to_kwargs()
    seed_ffn = FfnWindow(exp_dim=16, pos_anchor=2, neg_anchor=2).to_kwargs()

    for site in get_adapter(model).layer_sites(host):
        attn = VLPSoftmax(**seed_attn, layer=site.layer_idx, device='cpu',
                          profile_path=host._tmp, profile_dims=[0], keys=[], **site.keys)
        ffn = VLPSilu(**seed_ffn, layer=site.layer_idx, device='cpu',
                      profile_path=host._tmp, profile_dims=[0], keys=[], **site.keys)
        host.nonlinear_sites.append({
            'layer': site.layer_idx, 'keys': dict(site.keys),
            'attention': attn, 'ffn': ffn,
        })
    return host, model


class TestSiteRegistry:
    def test_registry_covers_every_layer(self, patched):
        host, model = patched
        assert len(host.nonlinear_sites) == model.config.num_hidden_layers

    def test_by_layer_lookup_is_dense(self, patched):
        host, model = patched
        by_layer = host.nonlinear_sites_by_layer()
        assert sorted(by_layer) == list(range(model.config.num_hidden_layers))

    def test_duplicate_layer_indices_are_rejected(self, patched):
        host, _ = patched
        host.nonlinear_sites.append(dict(host.nonlinear_sites[0]))
        with pytest.raises(RuntimeError, match="not unique"):
            host.nonlinear_sites_by_layer()

    def test_empty_registry_explains_the_legacy_path(self, patched):
        host, _ = patched
        host.nonlinear_sites = []
        with pytest.raises(RuntimeError, match="MUGI_USE_LEGACY_PATCH"):
            host.nonlinear_sites_by_layer()


class TestApplyAssignment:
    def test_applies_every_layer_and_site(self, patched):
        host, model = patched
        n_layers = model.config.num_hidden_layers
        a = WindowAssignment.uniform(
            n_layers,
            attention=SoftmaxWindow(exp_dim=8, anchor=-2),
            ffn=FfnWindow(exp_dim=8, pos_anchor=-1, neg_anchor=-1),
        )
        assert apply_assignment(host, a) == 2 * n_layers

    def test_per_layer_windows_actually_differ_on_the_model(self, patched):
        host, model = patched
        n_layers = model.config.num_hidden_layers

        a = WindowAssignment()
        for layer in range(n_layers):
            a.set(layer, ATTENTION, SoftmaxWindow(exp_dim=8, anchor=layer - 4))
        apply_assignment(host, a, sites=[ATTENTION])

        by_layer = host.nonlinear_sites_by_layer()
        anchors = [by_layer[i][ATTENTION].max_exp for i in range(n_layers)]
        assert anchors == [i - 4 for i in range(n_layers)]

        luts = [by_layer[i][ATTENTION].lut for i in range(n_layers)]
        assert not torch.equal(luts[0], luts[-1])

    def test_lut_matches_a_fresh_object_after_apply(self, patched, tmp_path):
        from custom_nonlinear.custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax

        host, _ = patched
        target = SoftmaxWindow(exp_dim=8, anchor=-3, anchor_side="min")
        apply_assignment(host, WindowAssignment({(0, ATTENTION): target}), sites=[ATTENTION])

        fresh = VLPSoftmax(**target.to_kwargs(), layer=0, device='cpu',
                           profile_path=str(tmp_path) + '/', profile_dims=[0], keys=[])
        assert torch.equal(host.nonlinear_sites_by_layer()[0][ATTENTION].lut, fresh.lut)

    def test_unknown_layer_is_rejected(self, patched):
        host, model = patched
        far = model.config.num_hidden_layers + 5
        a = WindowAssignment({(far, ATTENTION): SoftmaxWindow(exp_dim=8, anchor=0)})
        with pytest.raises(KeyError, match=f"layer {far}"):
            apply_assignment(host, a)

    def test_applying_nothing_is_an_error(self, patched):
        host, _ = patched
        a = WindowAssignment({(0, ATTENTION): SoftmaxWindow(exp_dim=8, anchor=0)})
        with pytest.raises(ValueError, match="applied 0 windows"):
            apply_assignment(host, a, sites=[FFN])

    def test_repeated_apply_is_cheap_and_stable(self, patched):
        host, model = patched
        a = WindowAssignment.uniform(
            model.config.num_hidden_layers,
            attention=SoftmaxWindow(exp_dim=8, anchor=1),
        )
        apply_assignment(host, a, sites=[ATTENTION])
        first = host.nonlinear_sites_by_layer()[0][ATTENTION].lut.clone()
        for _ in range(20):
            apply_assignment(host, a, sites=[ATTENTION])
        assert torch.equal(host.nonlinear_sites_by_layer()[0][ATTENTION].lut, first)


class TestSetProfiling:
    def test_toggles_every_collector(self, patched):
        host, model = patched
        n = set_profiling(host, False)
        assert n == 2 * model.config.num_hidden_layers
        assert all(e[s].profiling_enabled is False
                   for e in host.nonlinear_sites for s in (ATTENTION, FFN))

        set_profiling(host, True)
        assert all(e[s].profiling_enabled is True
                   for e in host.nonlinear_sites for s in (ATTENTION, FFN))

    def test_disabled_run_writes_no_profile_files(self, patched, tmp_path):
        host, model = patched
        set_profiling(host, False)

        before = set(tmp_path.rglob('*.pt'))
        ids = torch.randint(0, model.config.vocab_size, (1, 8))
        with torch.no_grad():
            model(input_ids=ids, labels=ids, use_cache=False)
        assert set(tmp_path.rglob('*.pt')) == before

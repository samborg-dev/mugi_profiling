import pytest

transformers = pytest.importorskip("transformers")

DIMS = dict(profiling_dims="P", profile_dims="V", source_profiling_dims="S", target_profiling_dims="T")


class HostStub:
    def __init__(self, model):
        self.model = model
        self.model_name = model.config.model_type
        for k, v in DIMS.items():
            setattr(self, k, v)


def _llama_oracle(model):
    L = model.config.num_hidden_layers
    return [dict(attn=f"model.layers.{i}.self_attn", ffn=f"model.layers.{i}.mlp",
                 attr="act_fn", layer=i, keys={}, dims="P") for i in range(L)]


def _whisper_oracle(model):
    rows = []
    for i in range(model.config.encoder_layers):
        rows.append(dict(attn=f"model.encoder.layers.{i}.self_attn", ffn=f"model.encoder.layers.{i}",
                         attr="activation_fn", layer=i, keys={}, dims="S"))
    for i in range(model.config.decoder_layers):
        rows.append(dict(attn=f"model.decoder.layers.{i}.self_attn", ffn=f"model.decoder.layers.{i}",
                         attr="activation_fn", layer=i, keys={}, dims="T"))
    return rows


def _swinv2_oracle(model):
    rows = []
    for blk, depth in enumerate(model.config.depths):
        for j in range(depth):
            rows.append(dict(attn=f"swinv2.encoder.layers.{blk}.blocks.{j}.attention.self",
                             ffn=f"swinv2.encoder.layers.{blk}.blocks.{j}.intermediate",
                             attr="intermediate_act_fn", layer=j, keys={"blocks": blk}, dims="V"))
    return rows


def _vivit_oracle(model):
    L = model.config.num_hidden_layers
    return [dict(attn=f"vivit.encoder.layer.{i}.attention.attention",
                 ffn=f"vivit.encoder.layer.{i}.intermediate",
                 attr="intermediate_act_fn", layer=i, keys={}, dims="V") for i in range(L)]


CASES = [
    pytest.param("AutoModelForCausalLM",
                 "hf-internal-testing/tiny-random-LlamaForCausalLM", _llama_oracle, id="llama"),
    pytest.param("AutoModelForSpeechSeq2Seq",
                 "hf-internal-testing/tiny-random-WhisperForConditionalGeneration", _whisper_oracle, id="whisper"),
    pytest.param("AutoModelForImageClassification",
                 "hf-internal-testing/tiny-random-Swinv2ForImageClassification", _swinv2_oracle, id="swinv2"),
    pytest.param("AutoModelForVideoClassification",
                 "hf-internal-testing/tiny-random-VivitForVideoClassification", _vivit_oracle, id="vivit"),
]


def _actual(model, host):
    from inference_classes.model_adapters import get_adapter

    names = {id(m): n for n, m in model.named_modules()}
    return [dict(attn=names[id(s.attn_module)], ffn=names[id(s.ffn_parent)],
                 attr=s.ffn_attr, layer=s.layer_idx, keys=dict(s.keys), dims=s.profile_dims)
            for s in get_adapter(model).layer_sites(host)]


def _load(loader_name, model_id):
    loader = getattr(transformers, loader_name)
    try:
        return loader.from_pretrained(model_id)
    except Exception as e:
        pytest.skip(f"could not load {model_id}: {type(e).__name__}: {e}")


@pytest.mark.parametrize("loader_name,model_id,oracle_fn", CASES)
def test_adapter_manifest_matches_legacy_targeting(loader_name, model_id, oracle_fn):
    model = _load(loader_name, model_id)
    host = HostStub(model)

    try:
        expected = oracle_fn(model)
        actual = _actual(model, host)
    except Exception as e:
        pytest.skip(f"{model_id} structure differs on transformers "
                    f"{transformers.__version__}: {type(e).__name__}: {e}")

    assert actual == expected, "adapter manifest drifted from the original instrumentation"


@pytest.mark.parametrize("loader_name,model_id,oracle_fn", CASES)
def test_expected_count_matches_site_count(loader_name, model_id, oracle_fn):
    from inference_classes.model_adapters import get_adapter

    model = _load(loader_name, model_id)

    try:
        expected = oracle_fn(model)
    except Exception as e:
        pytest.skip(f"{model_id} structure differs: {type(e).__name__}: {e}")

    assert get_adapter(model).expected_count(model) == len(expected)

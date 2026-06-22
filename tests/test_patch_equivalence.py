import sys

DIMS = dict(profiling_dims="P", profile_dims="V", source_profiling_dims="S", target_profiling_dims="T")


class HostStub:
    """Stands in for InferenceModel: only the attributes the adapters read."""
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
    ("AutoModelForCausalLM",            ["hf-internal-testing/tiny-random-LlamaForCausalLM"],               _llama_oracle),
    ("AutoModelForSpeechSeq2Seq",       ["hf-internal-testing/tiny-random-WhisperForConditionalGeneration"], _whisper_oracle),
    ("AutoModelForImageClassification", ["hf-internal-testing/tiny-random-Swinv2ForImageClassification"],    _swinv2_oracle),
    ("AutoModelForVideoClassification", ["hf-internal-testing/tiny-random-VivitForVideoClassification"], _vivit_oracle),
]


def _actual(model, host):
    from inference_classes.model_adapters import get_adapter
    names = {id(m): n for n, m in model.named_modules()}
    rows = []
    for s in get_adapter(model).layer_sites(host):
        rows.append(dict(attn=names[id(s.attn_module)], ffn=names[id(s.ffn_parent)],
                         attr=s.ffn_attr, layer=s.layer_idx, keys=dict(s.keys), dims=s.profile_dims))
    return rows


def main():
    try:
        import torch  
        import transformers
    except ModuleNotFoundError as e:
        print(f"SKIPPED: {e.name!r} not installed - run on the cluster (needs torch + transformers).")
        print("nothing was proven here.")
        return 0

    from inference_classes.model_adapters import get_adapter

    proven, failures = 0, []
    for loader_name, model_ids, oracle_fn in CASES:
        loader = getattr(transformers, loader_name)
        model = model_id = last = None
        for cand in model_ids:
            try:
                model, model_id = loader.from_pretrained(cand), cand
                break
            except Exception as e:  
                last = e
        if model is None:
            print(f"SKIP  {model_ids} (could not load: {type(last).__name__})")
            continue

        host = HostStub(model)
        try:
            expected_rows = oracle_fn(model)
            actual_rows = _actual(model, host)
        except Exception as e:
            print(f"SKIP  {model_id} (structure differs on this transformers version: {type(e).__name__}: {e})")
            continue

        ok = (actual_rows == expected_rows
              and get_adapter(model).expected_count(model) == len(expected_rows))
        if ok:
            print(f"PASS  {model_id}  ({len(actual_rows)} sites)")
            proven += 1
        else:
            failures.append(model_id)
            print(f"FAIL  {model_id}: adapter manifest differs from the original targeting")

    if failures:
        print(f"\n{len(failures)} architecture(s) DRIFTED from the original instrumentation.")
        return 1
    if proven == 0:
        print("\nnothing proven (all skipped). Run where the models + stack are available.")
        return 1
    print(f"\nadapter path reproduces the original instrumentation exactly on {proven} architecture(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

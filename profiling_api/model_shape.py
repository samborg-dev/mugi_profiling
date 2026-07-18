from typing import Tuple


def _get(config, *names, default=None):
    for n in names:
        if hasattr(config, n) and getattr(config, n) is not None:
            return getattr(config, n)
    return default


def _llm_shape(config) -> dict:
    heads = _get(config, "num_attention_heads")
    return {
        "dim": _get(config, "hidden_size"),
        "hidden_dim": _get(config, "intermediate_size"),
        "heads": heads,
        "kv_heads": _get(config, "num_key_value_heads", default=heads),
        "layers": _get(config, "num_hidden_layers"),
        "vocab_size": _get(config, "vocab_size"),
    }


def _whisper_shape(config) -> dict:
    heads = _get(config, "encoder_attention_heads", "num_attention_heads")
    return {
        "dim": _get(config, "d_model", "hidden_size"),
        "hidden_dim": _get(config, "encoder_ffn_dim", "decoder_ffn_dim"),
        "heads": heads,
        "kv_heads": heads,
        "layers": (_get(config, "encoder_layers", default=0)
                   + _get(config, "decoder_layers", default=0)),
        "vocab_size": _get(config, "vocab_size"),
    }


def _swinv2_shape(config) -> dict:
    embed = _get(config, "embed_dim", "hidden_size")
    depths = _get(config, "depths", default=[]) or []
    num_heads = _get(config, "num_heads", default=[]) or []
    mlp_ratio = _get(config, "mlp_ratio", default=4.0)
    return {
        "dim": embed,
        "hidden_dim": int(embed * mlp_ratio) if embed else None,
        "heads": num_heads[0] if num_heads else None,
        "kv_heads": num_heads[0] if num_heads else None,
        "layers": sum(depths) if depths else _get(config, "num_hidden_layers"),
        "vocab_size": _get(config, "num_labels", default=0),
    }


def _vivit_shape(config) -> dict:
    heads = _get(config, "num_attention_heads")
    return {
        "dim": _get(config, "hidden_size"),
        "hidden_dim": _get(config, "intermediate_size"),
        "heads": heads,
        "kv_heads": heads,
        "layers": _get(config, "num_hidden_layers"),
        "vocab_size": _get(config, "num_labels", default=0),
    }


_SHAPERS = {
    "llama": (_llm_shape, True),
    "whisper": (_whisper_shape, False),
    "swinv2": (_swinv2_shape, False),
    "vivit": (_vivit_shape, False),
}


def model_shape_fields(model) -> Tuple[dict, bool]:
    config = getattr(model, "config", None)
    mtype = getattr(config, "model_type", None)
    if mtype in _SHAPERS:
        extractor, is_llm = _SHAPERS[mtype]
        return extractor(config), is_llm
    fields = _llm_shape(config)
    is_llm = all(fields.get(k) is not None for k in ("dim", "hidden_dim", "heads", "layers"))
    if not is_llm:
        raise ValueError(
            f"No archx model-shape mapping for model_type={mtype!r}. "
            f"Add one in profiling_api/model_shape.py."
        )
    return fields, is_llm

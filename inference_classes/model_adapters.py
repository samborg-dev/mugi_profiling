from dataclasses import dataclass, field
from typing import Callable

from custom_nonlinear.custom_eager import LlamaEager, VivitEager, WhisperEager
from custom_nonlinear.custom_forward import (
    llama_forward,
    swin_forward,
    vivit_forward,
    whisper_forward,
)


@dataclass
class LayerSite:
    """One layer's instrumentation points (attention + FFN are always paired per layer)."""
    layer_idx: int                 # value passed as `layer=` to the collector
    device: object                 # next(layer.parameters()).device
    profile_dims: object           # which host.*_profiling_dims to record along
    attn_module: object            # module whose `.forward` gets patched
    forward_builder: Callable      # nonlinear_object -> forward fn (wraps eager if needed)
    ffn_parent: object             # module that owns the activation attribute
    ffn_attr: str                  # attribute name to overwrite with the FFN collector
    set_device: bool               # reproduce the old `if i == 0: self.device = layer_device`
    keys: dict = field(default_factory=dict)   # extra collector kwargs (e.g. blocks=i for swin)


def _device_of(module):
    return next(module.parameters()).device


class ModelAdapter:
    """Base class. Subclasses set match keys and implement `layer_sites`."""
    model_types: set = set()       # matches model.config.model_type
    name_keys: tuple = ()          # fallback substrings matched against the model name

    @classmethod
    def matches(cls, model) -> bool:
        mtype = getattr(getattr(model, "config", None), "model_type", "") or ""
        if mtype in cls.model_types:
            return True
        archs = getattr(getattr(model, "config", None), "architectures", None) or []
        name = (mtype + " " + " ".join(archs)).lower()
        return any(k in name for k in cls.name_keys)

    def layer_sites(self, host):
        """Yield a LayerSite per layer. `host` is the InferenceModel (for profiling dims)."""
        raise NotImplementedError

    def expected_count(self, model) -> int:
        """Layer count derived INDEPENDENTLY from the model config (not by walking modules).

        patch_model cross-checks the number of layers it actually instrumented against this. A
        mismatch (or zero) means the wrong/partial set of layers was profiled -> hard error,
        never a silent drift in the window config that downstream accuracy depends on.
        """
        raise NotImplementedError


class LlamaAdapter(ModelAdapter):
    model_types = {"llama"}
    name_keys = ("llama",)

    def layer_sites(self, host):
        for i, layer in enumerate(host.model.model.layers):
            yield LayerSite(
                layer_idx=i,
                device=_device_of(layer),
                profile_dims=host.profiling_dims,
                attn_module=layer.self_attn,
                forward_builder=lambda nl: llama_forward(LlamaEager(nonlinear_object=nl)),
                ffn_parent=layer.mlp,
                ffn_attr="act_fn",
                set_device=(i == 0),
            )

    def expected_count(self, model):
        return model.config.num_hidden_layers


class WhisperAdapter(ModelAdapter):
    model_types = {"whisper"}
    name_keys = ("whisper",)

    def layer_sites(self, host):
        # encoder then decoder; they record along different profiling dims
        for stack, dims in (
            (host.model.model.encoder.layers, host.source_profiling_dims),
            (host.model.model.decoder.layers, host.target_profiling_dims),
        ):
            for i, layer in enumerate(stack):
                yield LayerSite(
                    layer_idx=i,
                    device=_device_of(layer),
                    profile_dims=dims,
                    attn_module=layer.self_attn,
                    forward_builder=lambda nl: whisper_forward(WhisperEager(nonlinear_object=nl)),
                    ffn_parent=layer,
                    ffn_attr="activation_fn",
                    set_device=(i == 0),
                )

    def expected_count(self, model):
        return model.config.encoder_layers + model.config.decoder_layers


class Swinv2Adapter(ModelAdapter):
    model_types = {"swinv2"}
    name_keys = ("swinv2",)

    def layer_sites(self, host):
        for block_idx, block in enumerate(host.model.swinv2.encoder.layers):
            for j, layer in enumerate(block.blocks):
                yield LayerSite(
                    layer_idx=j,
                    device=_device_of(layer),
                    profile_dims=host.profile_dims,
                    attn_module=layer.attention.self,
                    forward_builder=lambda nl: swin_forward(nl),   # swin has no eager wrapper
                    ffn_parent=layer.intermediate,
                    ffn_attr="intermediate_act_fn",
                    set_device=(block_idx == 0),
                    keys={"blocks": block_idx},
                )

    def expected_count(self, model):
        return sum(model.config.depths)


class VivitAdapter(ModelAdapter):
    model_types = {"vivit"}
    name_keys = ("vivit",)

    def layer_sites(self, host):
        for i, layer in enumerate(host.model.vivit.encoder.layer):
            yield LayerSite(
                layer_idx=i,
                device=_device_of(layer),
                profile_dims=host.profile_dims,
                attn_module=layer.attention.attention,
                forward_builder=lambda nl: vivit_forward(VivitEager(nonlinear_object=nl)),
                ffn_parent=layer.intermediate,
                ffn_attr="intermediate_act_fn",
                set_device=(i == 0),
            )

    def expected_count(self, model):
        return model.config.num_hidden_layers


# Registration order = match priority. New architectures: add an adapter and append it here.
ADAPTERS = [LlamaAdapter(), WhisperAdapter(), Swinv2Adapter(), VivitAdapter()]


def get_adapter(model) -> ModelAdapter:
    """Return the adapter that handles `model`, matched on its config (not the model name)."""
    for adapter in ADAPTERS:
        if adapter.matches(model):
            return adapter
    mtype = getattr(getattr(model, "config", None), "model_type", "?")
    raise ValueError(
        f"No ModelAdapter registered for model_type={mtype!r}. "
        f"Add a ModelAdapter subclass in inference_classes/model_adapters.py."
    )

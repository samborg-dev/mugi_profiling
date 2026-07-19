from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class DatasetSpec:
    name: Optional[str] = None
    hf_path: Optional[str] = None
    config: Optional[str] = None
    split: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DatasetSpec":
        d = d or {}
        return cls(
            name=d.get("name"),
            hf_path=d.get("hf_path"),
            config=d.get("config"),
            split=d.get("split"),
        )


@dataclass
class ProfileConfig:
    model_id: str = ""
    modality: str = "nlp"
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    n_samples: int = 1
    batch_size: int = 1
    output_dir: str = "output"

    onnx_path: Optional[str] = None
    export_task: str = "auto"
    export_opset: Optional[int] = None

    window_size: int = 8
    exp_dim: int = 16
    lut_window_size: int = 32
    window_strategy: str = "topk"
    window_criterion: str = "auto"
    window_calibration: Optional[str] = None

    dtype: str = "FP16"
    quantization: str = "None"
    kv_quant_dim: str = "token"
    profile_targets: tuple = ("nonlinear_inputs",)

    activation_bitwidth: int = 16
    weight_bitwidth: int = 4
    noc_stationary: str = "os"
    node_stationary: str = "os"
    lut_height: int = 8
    lut_width: int = 12
    window_width: int = 8
    cycles: int = 8
    prefill_seq_len: int = 64
    max_seq_len: int = 512
    array_config: str = "128x8"
    node_topology: str = "single_node"
    variant: str = "lut"
    archx_workload_name: Optional[str] = None
    workload_format: str = "nested"          # nested | flat (which layout archx consumes)
    emit_onnx: bool = True                    # export ONNX and convert it to the archx workload
    early_termination_cycles: dict = field(default_factory=lambda: {
        "proj_avg_early_termination_cycles": 8,
        "ffn_avg_early_termination_cycles": 8,
        "k_avg_early_termination_cycles": 8,
        "v_avg_early_termination_cycles": 8,
        "nonlinear_avg_early_termination_cycles": 8,
        "default_avg_early_termination_cycles": 8,
    })

    model_dict: dict = field(default_factory=dict)
    nonlinear_dict: dict = field(default_factory=dict)
    parameter_dict: dict = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return self.model_id

    @classmethod
    def from_configs(cls, model_dict: dict, nonlinear_dict: dict,
                     parameter_dict: dict, **overrides) -> "ProfileConfig":
        model = model_dict.get("model", {})
        params = model_dict.get("parameters", {})
        cfg = cls(
            model_id=model.get("name", ""),
            modality=model.get("modality", "nlp"),
            dataset=DatasetSpec.from_dict(model_dict.get("dataset")),
            n_samples=parameter_dict.get("n_samples", 1),
            batch_size=params.get("batch_size", 1),
            model_dict=model_dict,
            nonlinear_dict=nonlinear_dict,
            parameter_dict=parameter_dict,
        )
        if overrides:
            cfg = replace(cfg, **overrides)
        return cfg

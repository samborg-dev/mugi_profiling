import os
import re

import yaml

from profiling_api.window import WindowSizer
from profiling_api.model_shape import model_shape_fields


class ConfigEmitter:
    def __init__(self, cfg):
        self.cfg = cfg

    def emit(self, store) -> str:
        WindowSizer(self.cfg).size(store)

        from profile_distribution import create_nonlinear_config
        dist_path = os.path.join("distribution", store.model_name)
        nonlinear_dict = create_nonlinear_config(dist_path, store.model_name)

        out_dir = os.path.join(self.cfg.output_dir, "nonlinear_config", store.model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "nonlinear_config.yaml")
        with open(out_path, "w") as f:
            yaml.dump(nonlinear_dict, f, sort_keys=False)
        return out_path


class ArchxWorkloadEmitter:
    def __init__(self, cfg):
        self.cfg = cfg

    def workload_name(self) -> str:
        if self.cfg.archx_workload_name:
            return self.cfg.archx_workload_name
        short = self.cfg.model_id.rstrip("/").split("/")[-1].lower()
        return re.sub(r"[^a-z0-9]+", "_", short).strip("_")

    def build_configuration(self, model):
        cfg = self.cfg
        fields, is_llm = model_shape_fields(model)
        configuration = {
            "architecture": "mugi",
            **fields,
            "batch_size": cfg.batch_size,
            "prefill_seq_len": cfg.prefill_seq_len,
            "max_seq_len": cfg.max_seq_len,
            "activation_bitwidth": cfg.activation_bitwidth,
            "weight_bitwidth": cfg.weight_bitwidth,
            "noc_stationary": cfg.noc_stationary,
            "node_stationary": cfg.node_stationary,
            "lut_height": cfg.lut_height,
            "lut_width": cfg.lut_width,
            "window_width": cfg.window_width,
            "cycles": cfg.cycles,
            **cfg.early_termination_cycles,
        }
        return configuration, is_llm

    def emit(self, model) -> str:
        cfg = self.cfg
        configuration, is_llm = self.build_configuration(model)
        name = self.workload_name()
        workload = {"workload": {name: {"configuration": configuration}}}

        out_dir = os.path.join(
            cfg.output_dir, "archx", "workload", name,
            f"{cfg.variant}_{cfg.array_config}",
            f"max_seq_len_{cfg.max_seq_len}", f"batch_size_{cfg.batch_size}",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "workload.yaml")

        header = ""
        if not is_llm:
            header = (
                f"# archx-structural-only: model_type={getattr(model.config, 'model_type', '?')!r} "
                f"is not LLM-shaped; archx cannot cost this until a matching performance model "
                f"exists. Schema-correct for inspection only.\n"
            )
        with open(out_path, "w") as f:
            if header:
                f.write(header)
            yaml.dump(workload, f, default_flow_style=False)
        return out_path

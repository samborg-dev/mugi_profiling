"""Quick local demo of the profiling API — runs with just torch (no model download, no GPU).

It exercises the real code on stub/synthetic inputs so you can see the two artifacts and the
equivalence checks without the cluster. Delete this file whenever; it is only a smoke demo.

    python demo_profiling_api.py
"""

import os
import shutil
import types

import torch
import yaml

from profiling_api.config import ProfileConfig
from profiling_api.window import WindowSizer
from profiling_api.emit import ConfigEmitter, ArchxWorkloadEmitter
from profile_distribution import profile_tensor

TMP = "output/_demo"
ARCHX_REF = (r"C:/Users/samue/Desktop/dev/Unary/archx/zoo/llm/mugi/workload/generated/"
             r"llama_2_13b/early_termination_128x8/max_seq_len_512/batch_size_1/workload.yaml")


def hr(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def demo_window_sizer():
    hr("1) WindowSizer — generic vs the Llama2_7B calibration")
    leaf = f"{TMP}/profile/softmax/pre_softmax/exp_dist/layer_0"
    os.makedirs(leaf, exist_ok=True)
    tp = f"{leaf}/seq_len_2.pt"
    t = torch.zeros(32)
    t[14:22] = torch.tensor([5., 20., 50., 80., 60., 30., 10., 4.])
    torch.save(t, tp)

    generic = WindowSizer(ProfileConfig(model_id="x"))._size_tensor(tp)
    calib = WindowSizer(ProfileConfig(model_id="x", window_calibration="Llama2_7B"))._size_tensor(tp)
    print("generic  ->", {k: generic[k] for k in ("max_exp", "min_exp", "cluster")})
    print("calibrated ->", {k: calib[k] for k in ("max_exp", "min_exp", "cluster")})
    print("calibration == original profile_tensor:", calib == profile_tensor(tp))


def demo_nonlinear_config():
    hr("2) ConfigEmitter — synthetic profiling -> nonlinear_config.yaml")
    model_name = "democo/demomodel"
    base = f"profile/{model_name}/vlp_softmax_vlp_silu"
    for nl in ("softmax", "silu"):
        for layer in (0, 1):
            d = f"{base}/{nl}/pre_{nl}/exp_dist/layer_{layer}"
            os.makedirs(d, exist_ok=True)
            t = torch.zeros(32)
            t[13 + layer:21 + layer] = torch.tensor([5., 20., 50., 80., 60., 30., 10., 4.])
            torch.save(t, f"{d}/seq_len_2.pt")

    class Store:
        pass
    store = Store()
    store.model_name = model_name
    store.root = f"profile/{model_name}"
    def tensor_dirs():
        from profile_distribution import loop_through_subdirs
        r = loop_through_subdirs(store.root)
        return r if isinstance(r, list) else [r]
    store.tensor_dirs = tensor_dirs

    out = ConfigEmitter(ProfileConfig(model_id=model_name, output_dir=TMP)).emit(store)
    print("wrote:", out, "\n")
    print(open(out).read())
    shutil.rmtree("profile/democo", ignore_errors=True)
    shutil.rmtree("distribution/democo", ignore_errors=True)


def demo_archx_workload():
    hr("3) ArchxWorkloadEmitter — model config -> archx workload.yaml")
    ns = types.SimpleNamespace(model_type="llama", hidden_size=5120, intermediate_size=13824,
                               num_attention_heads=40, num_key_value_heads=40,
                               num_hidden_layers=40, vocab_size=32000)
    cfg = ProfileConfig(
        model_id="meta-llama/Llama-2-13b-hf", archx_workload_name="llama_2_13b", output_dir=TMP,
        batch_size=1, prefill_seq_len=64, max_seq_len=512,
        early_termination_cycles={
            "proj_avg_early_termination_cycles": 7.04, "ffn_avg_early_termination_cycles": 7.04,
            "k_avg_early_termination_cycles": 7.266, "v_avg_early_termination_cycles": 7.8207,
            "nonlinear_avg_early_termination_cycles": 8, "default_avg_early_termination_cycles": 8},
    )
    out = ArchxWorkloadEmitter(cfg).emit(types.SimpleNamespace(config=ns))
    print("wrote:", out, "\n")
    print(open(out).read())

    if os.path.exists(ARCHX_REF):
        em = yaml.safe_load(open(out))["workload"]["llama_2_13b"]["configuration"]
        ref = yaml.safe_load(open(ARCHX_REF))["workload"]["llama_2_13b"]["configuration"]
        print("matches committed archx llama_2_13b workload:", em == ref)
    else:
        print("(archx repo not found at expected path — skipped equivalence diff)")


if __name__ == "__main__":
    try:
        demo_window_sizer()
        demo_nonlinear_config()
        demo_archx_workload()
        hr("done")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

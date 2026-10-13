import argparse
import csv
import importlib.util
import os
import sys
import types

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_eval import CELLS, FIELDS, measure, patch, token_hash

ATTRS = ('exp_dim', 'max_exp', 'min_exp', 'window_size', 'lut_build', 'mant_dim')
DIFF_FIELDS = ['kernel_mode', 'layer', 'rel_l2', 'max_abs', 'frac_diff', 'n_elems',
               'x_shift_at_max', 'main_at_max', 'ae_at_max', 'row_max_at_max']


def load_ae_kernel(path):
    stub = types.ModuleType('custom_nonlinear.custom_approx')
    stub.CustomSoftmax = object
    saved = sys.modules.get('custom_nonlinear.custom_approx')
    sys.modules['custom_nonlinear.custom_approx'] = stub
    try:
        spec = importlib.util.spec_from_file_location('ae_vlp_softmax', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules['custom_nonlinear.custom_approx'] = saved
        else:
            sys.modules.pop('custom_nonlinear.custom_approx', None)
    return mod.VLPSoftmax


def ae_view(ae_cls, obj):
    view = ae_cls.__new__(ae_cls)
    for name in ATTRS:
        setattr(view, name, getattr(obj, name))
    return view


def install(main_cls, ae_cls, state):
    main_nl = main_cls.nonlinear

    def nonlinear(self, attn_weights, dim=-1, dtype=torch.bfloat16):
        first = self.layer not in state['seen']
        out_main = main_nl(self, attn_weights, dim=dim, dtype=dtype) if (first or state['mode'] == 'main') else None
        out_ae = ae_view(ae_cls, self).nonlinear(attn_weights, dim=dim, dtype=dtype) if (first or state['mode'] == 'ae') else None
        if first:
            state['seen'].add(self.layer)
            a, b = out_main.float(), out_ae.float()
            d = (a - b).abs()
            i = int(d.argmax())
            xb = attn_weights.to(torch.bfloat16)
            rmax = xb.max(dim=dim, keepdim=True)[0].expand_as(xb)
            state['rows'].append({
                'kernel_mode': state['mode'], 'layer': self.layer,
                'rel_l2': ((a - b).norm() / a.norm()).item(), 'max_abs': d.max().item(),
                'frac_diff': (d > 0).float().mean().item(), 'n_elems': d.numel(),
                'x_shift_at_max': (xb - rmax).flatten()[i].item(),
                'main_at_max': a.flatten()[i].item(), 'ae_at_max': b.flatten()[i].item(),
                'row_max_at_max': rmax.flatten()[i].item(),
            })
            del a, b, d, xb, rmax
        return out_main if state['mode'] == 'main' else out_ae

    main_cls.nonlinear = nonlinear


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--ae_kernel', required=True)
    p.add_argument('--model_config', required=True)
    p.add_argument('--model_name', required=True)
    p.add_argument('--n_samples', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--out', required=True)
    p.add_argument('--diff_out', required=True)
    args = p.parse_args()

    out, diff_out, ae_kernel = (os.path.abspath(x) for x in (args.out, args.diff_out, args.ae_kernel))
    model_dict = yaml.safe_load(open(os.path.abspath(args.model_config)))
    os.chdir(args.repo)
    sys.path.insert(0, os.getcwd())
    from inference_classes.npl_inference import NLPModel
    from custom_nonlinear.custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax as MainVLPSoftmax
    AEVLPSoftmax = load_ae_kernel(ae_kernel)

    model_dict['model']['name'] = args.model_name
    model_dict['model']['modality'] = 'nlp'
    model_dict['parameters']['batch_size'] = args.batch_size
    nonlinear_dict = {'functions': {'vlp': {'attention': ['softmax'], 'ffn': ['silu']}}, 'params': {}}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    host = NLPModel(model_dict, nonlinear_dict, {'n_samples': args.n_samples}, device)

    print('Loading model', flush=True)
    host.load_model()
    host.load_streaming_dataset()
    host.process_dataset()
    raw = list(host.inputs)
    th = token_hash(raw)
    n_tokens = sum(int(ex['input_ids'].numel()) for ex in raw)
    host.batch_dataset()
    host.set_profiling_dims()

    state = {'mode': 'main', 'seen': set(), 'rows': []}
    install(MainVLPSoftmax, AEVLPSoftmax, state)
    print('Looping through configurations', flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for mode in ('main', 'ae'):
            state['mode'] = mode
            state['seen'] = set()
            ppl, wall, status = measure(host, lambda: patch(host, 'main', *CELLS['main']['softmax']))
            row = {'code': f'main+{mode}_kernel', 'batch_size': args.batch_size, 'cell': 'softmax',
                   'ppl': ppl, 'num_batches': host.num_batches if status == 'ok' else None,
                   'n_tokens': n_tokens, 'token_hash': th,
                   'total_loss': host.total_loss if status == 'ok' else None,
                   'wall_s': round(wall, 2), 'peak_mem_gb': None, 'status': status}
            writer.writerow(row)
            f.flush()
            print(row, flush=True)

    with open(diff_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=DIFF_FIELDS)
        writer.writeheader()
        writer.writerows(state['rows'])
    print('Saving results', flush=True)


if __name__ == '__main__':
    main()

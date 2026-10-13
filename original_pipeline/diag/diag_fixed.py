import argparse
import csv
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_eval import CELLS, FIELDS, measure, patch, token_hash
from diag_swap import ae_view, load_ae_kernel

MASK_THRESHOLD = -65500.0
STAT_FIELDS = ['kernel', 'layer', 'frac_neginf', 'frac_masked_finite', 'rows_with_neginf',
               'leaked_mass_mean', 'leaked_mass_max']
RUNS = [
    ('main', False, 'softmax', True),
    ('ae', False, 'softmax', True),
    ('main', True, 'softmax', False),
    ('ae', True, 'softmax', False),
    ('main', True, 'both', False),
]


def install(main_cls, ae_cls, state):
    main_nl = main_cls.nonlinear

    def nonlinear(self, attn_weights, dim=-1, dtype=torch.bfloat16):
        if state['kernel'] == 'main':
            out = main_nl(self, attn_weights, dim=dim, dtype=dtype)
        else:
            out = ae_view(ae_cls, self).nonlinear(attn_weights, dim=dim, dtype=dtype)
        masked = attn_weights <= MASK_THRESHOLD
        if state['stats'] and self.layer not in state['seen']:
            state['seen'].add(self.layer)
            neginf = torch.isneginf(attn_weights)
            leaked = out.float().masked_fill(~masked, 0.0).sum(dim=dim)
            state['rows'].append({
                'kernel': state['kernel'], 'layer': self.layer,
                'frac_neginf': neginf.float().mean().item(),
                'frac_masked_finite': (masked & ~neginf).float().mean().item(),
                'rows_with_neginf': neginf.any(dim=dim).float().mean().item(),
                'leaked_mass_mean': leaked.mean().item(),
                'leaked_mass_max': leaked.max().item(),
            })
            del neginf, leaked
        if state['fix']:
            fixed = out.float().masked_fill(masked, 0.0)
            out = (fixed / fixed.sum(dim=dim, keepdim=True)).to(out.dtype)
            del fixed
        return out

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
    p.add_argument('--stats_out', required=True)
    args = p.parse_args()

    out, stats_out, ae_kernel = (os.path.abspath(x) for x in (args.out, args.stats_out, args.ae_kernel))
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

    state = {'kernel': 'main', 'fix': False, 'stats': False, 'seen': set(), 'rows': []}
    install(MainVLPSoftmax, AEVLPSoftmax, state)
    print('Looping through configurations', flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for kernel, fix, cell, stats in RUNS:
            state.update(kernel=kernel, fix=fix, stats=stats, seen=set())
            ppl, wall, status = measure(host, lambda cell=cell: patch(host, 'main', *CELLS['main'][cell]))
            row = {'code': f"{kernel}_kernel_{'fixed' if fix else 'buggy'}", 'batch_size': args.batch_size,
                   'cell': cell, 'ppl': ppl, 'num_batches': host.num_batches if status == 'ok' else None,
                   'n_tokens': n_tokens, 'token_hash': th,
                   'total_loss': host.total_loss if status == 'ok' else None,
                   'wall_s': round(wall, 2), 'peak_mem_gb': None, 'status': status}
            writer.writerow(row)
            f.flush()
            print(row, flush=True)

    with open(stats_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=STAT_FIELDS)
        writer.writeheader()
        writer.writerows(state['rows'])
    print('Saving results', flush=True)


if __name__ == '__main__':
    main()

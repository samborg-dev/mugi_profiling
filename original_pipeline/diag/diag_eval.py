import argparse
import csv
import gc
import hashlib
import os
import sys
import time

import torch
import yaml

CELLS = {
    'ae': {
        'softmax': ({'exp_dim': 10, 'max_exp': 2, 'min_exp': 4, 'window_size': 256, 'lut_build': 'max'}, {}),
        'both': ({'exp_dim': 10, 'max_exp': 2, 'min_exp': 4, 'window_size': 256, 'lut_build': 'max'},
                 {'exp_dim': 10, 'max_pos_exp': 2, 'min_pos_exp': -4, 'window_size': 256, 'lut_build': 'max'}),
    },
    'main': {
        'softmax': ({'exp_dim': 10, 'max_exp': 2, 'min_exp': -7, 'window_size': 256, 'lut_build': 'max'}, {}),
        'both': ({'exp_dim': 10, 'max_exp': 2, 'min_exp': -7, 'window_size': 256, 'lut_build': 'max'},
                 {'exp_dim': 10, 'max_pos_exp': 2, 'max_neg_exp': 2, 'window_size': 256}),
    },
}

FIELDS = ['code', 'batch_size', 'cell', 'ppl', 'num_batches', 'n_tokens', 'token_hash',
          'total_loss', 'wall_s', 'peak_mem_gb', 'status']


def token_hash(raw):
    h = hashlib.sha256()
    for ex in raw:
        h.update(ex['input_ids'].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def peak_gb():
    if not torch.cuda.is_available():
        return 0.0
    return max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30


def reset_peak():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def profiling_off(host):
    for entry in getattr(host, 'nonlinear_sites', []):
        for obj in (entry.get('attention'), entry.get('ffn')):
            if obj is not None and hasattr(obj, 'profiling_enabled'):
                obj.profiling_enabled = False


def patch(host, code, attn, ffn):
    patch_ffn = bool(ffn)
    if code == 'ae':
        host.patch_model('vlp', attention_parameters=dict(attn), ffn_parameters=dict(ffn),
                         patch_attention=True, patch_ffn=patch_ffn)
    else:
        host.patch_model('vlp', attention_parameters=dict(attn), ffn_parameters=dict(ffn),
                         patch_attention=True, patch_ffn=patch_ffn, run_eval=False)
        profiling_off(host)
        host.run_inference()


def measure(host, fn):
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        fn()
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return None, time.perf_counter() - t0, 'OOM'
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return host.metric, time.perf_counter() - t0, 'ok'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', choices=['ae', 'main'], required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--model_config', required=True)
    p.add_argument('--model_name', required=True)
    p.add_argument('--n_samples', type=int, default=8)
    p.add_argument('--batch_sizes', default='8,4,2')
    p.add_argument('--cells', default='softmax,both')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = os.path.abspath(args.out)
    model_dict = yaml.safe_load(open(os.path.abspath(args.model_config)))
    os.chdir(args.repo)
    sys.path.insert(0, os.getcwd())
    from inference_classes.npl_inference import NLPModel

    model_dict['model']['name'] = args.model_name
    model_dict['model']['modality'] = 'nlp'
    nonlinear_dict = {'functions': {'vlp': {'attention': ['softmax'], 'ffn': ['silu']}}, 'params': {}}
    parameter_dict = {'n_samples': args.n_samples}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    host = NLPModel(model_dict, nonlinear_dict, parameter_dict, device)

    print('Loading model', flush=True)
    host.load_model()
    host.load_streaming_dataset()
    host.process_dataset()
    raw = list(host.inputs)
    if len(raw) != args.n_samples:
        raise SystemExit(f'expected {args.n_samples} samples, got {len(raw)}')
    th = token_hash(raw)
    n_tokens = sum(int(ex['input_ids'].numel()) for ex in raw)
    host.set_profiling_dims()
    print('Looping through configurations', flush=True)

    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    plan = [(bs, 'baseline', None) for bs in batch_sizes]
    plan += [(bs, c, CELLS[args.code][c]) for bs in batch_sizes for c in args.cells.split(',')]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        current_bs = None
        for bs, cell, params in plan:
            if bs != current_bs:
                host.inputs = list(raw)
                host.batch_size = bs
                host.batch_dataset()
                current_bs = bs
            if params is None:
                fn = host.run_inference
            else:
                fn = lambda params=params: patch(host, args.code, *params)
            ppl, wall, status = measure(host, fn)
            row = {'code': args.code, 'batch_size': bs, 'cell': cell, 'ppl': ppl,
                   'num_batches': host.num_batches if status == 'ok' else None,
                   'n_tokens': n_tokens, 'token_hash': th,
                   'total_loss': host.total_loss if status == 'ok' else None,
                   'wall_s': round(wall, 2), 'peak_mem_gb': round(peak_gb(), 1), 'status': status}
            writer.writerow(row)
            f.flush()
            print(row, flush=True)
    print('Saving results', flush=True)


if __name__ == '__main__':
    main()

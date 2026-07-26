import argparse
import os
import statistics
import sys

import yaml

from profiling_api.config import ProfileConfig
from profiling_api.evaluate import WindowEvalHarness
from profiling_api.pipeline import ModelLoader
from profiling_api.windows import FfnWindow, SoftmaxWindow


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure LUT-window evaluation cost and the perplexity noise floor.")
    p.add_argument('--model_config', required=True)
    p.add_argument('--nonlinear_config', required=True)
    p.add_argument('--parameter_config', required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--seq_len', type=int, default=None)
    p.add_argument('--exp_dim', type=int, default=16)
    p.add_argument('--anchor', type=int, default=1)
    p.add_argument('--anchor_side', choices=['max', 'min'], default='max')
    p.add_argument('--group_size', type=int, default=32)
    p.add_argument('--no_baseline', action='store_true')
    p.add_argument('--out', default='output/window_search/cost_and_noise.csv')
    return p.parse_args(argv)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_config(args):
    model_dict = load_yaml(args.model_config)
    nonlinear_dict = load_yaml(args.nonlinear_config)
    parameter_dict = load_yaml(args.parameter_config)

    if args.n_samples is not None:
        parameter_dict['n_samples'] = args.n_samples

    return ProfileConfig.from_configs(model_dict, nonlinear_dict, parameter_dict)


def summarise(results):
    ppls = [r.ppl for r in results]
    walls = [r.wall_s for r in results]
    finite = [p for p in ppls if p == p]

    spread = (max(finite) - min(finite)) if len(finite) > 1 else 0.0
    stdev = statistics.stdev(finite) if len(finite) > 1 else 0.0

    return {
        'repeats': len(results),
        'ppl_mean': statistics.fmean(finite) if finite else float('nan'),
        'ppl_spread': spread,
        'ppl_stdev': stdev,
        'wall_s_median': statistics.median(walls),
        'wall_s_total': sum(walls),
        'apply_ms_median': statistics.median(r.apply_ms for r in results),
        'n_tokens': results[0].n_tokens,
        'num_batches': results[0].num_batches,
    }


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)

    model = ModelLoader().load(cfg)
    if args.seq_len is not None:
        print(f"note: --seq_len is recorded but process_dataset already ran at "
              f"max_length={getattr(model, 'max_length', '?')}", file=sys.stderr)

    model.model.eval()

    harness = WindowEvalHarness(
        model,
        seed_attention=SoftmaxWindow(exp_dim=args.exp_dim, anchor=args.anchor,
                                     anchor_side=args.anchor_side, group_size=args.group_size),
        seed_ffn=FfnWindow(exp_dim=args.exp_dim, pos_anchor=args.anchor,
                           neg_anchor=args.anchor, group_size=args.group_size),
    )
    harness.setup(measure_baseline=not args.no_baseline)

    if harness.baseline_ppl is not None:
        print(f"baseline (unpatched) ppl: {harness.baseline_ppl:.6f}")

    assignment = harness.uniform()
    print(f"assignment digest: {assignment.digest()}  "
          f"input hash: {harness.input_hash}  layers: {harness.n_layers}")

    results = []
    for i in range(args.repeats):
        r = harness.evaluate(assignment)
        results.append(r)
        print(f"  repeat {i + 1}/{args.repeats}: ppl={r.ppl:.6f} "
              f"wall={r.wall_s:.2f}s apply={r.apply_ms:.2f}ms")

    stats = summarise(results)
    path = harness.write_csv(args.out)

    print()
    print(f"ppl mean          {stats['ppl_mean']:.6f}")
    print(f"ppl spread        {stats['ppl_spread']:.6g}")
    print(f"ppl stdev         {stats['ppl_stdev']:.6g}")
    print(f"wall/eval median  {stats['wall_s_median']:.2f}s")
    print(f"apply/eval median {stats['apply_ms_median']:.2f}ms")
    print(f"tokens/eval       {stats['n_tokens']}")
    print(f"wrote             {os.path.abspath(path)}")

    summary_path = os.path.splitext(path)[0] + '_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.safe_dump({**stats,
                        'baseline_ppl': harness.baseline_ppl,
                        'assignment_digest': assignment.digest(),
                        'input_hash': harness.input_hash,
                        'provenance': results[0].provenance}, f, sort_keys=True)
    print(f"wrote             {os.path.abspath(summary_path)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

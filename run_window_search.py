import argparse
import math
import os
import random
import statistics
import sys

import yaml

from profiling_api.config import ProfileConfig
from profiling_api.evaluate import WindowEvalHarness, timed_load
from profiling_api.pipeline import ModelLoader
from profiling_api.search import CandidateGrid, ProgressiveLayerSearch, SearchBudget
from profiling_api.seed import HistogramSeeder
from profiling_api.windows import DEFAULT_GROUP_SIZE, FfnWindow, SoftmaxWindow


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure LUT-window evaluation cost and the perplexity noise floor "
                    "(--mode noise), or run the automated per-layer window search "
                    "(--mode search).")
    p.add_argument('--model_config', required=True)
    p.add_argument('--nonlinear_config', required=True)
    p.add_argument('--parameter_config', required=True)
    p.add_argument('--mode', choices=['noise', 'search'], default='noise')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--seq_len', type=int, default=None)
    p.add_argument('--exp_dim', type=int, default=10)
    p.add_argument('--anchor', type=int, default=2)
    p.add_argument('--anchor_side', choices=['max', 'min'], default='max')
    p.add_argument('--group_size', type=int, default=DEFAULT_GROUP_SIZE)
    p.add_argument('--no_baseline', action='store_true')
    p.add_argument('--out', default='output/window_search/cost_and_noise.csv')

    p.add_argument('--noise_floor', type=float, default=None)
    p.add_argument('--noise_summary', default=None)
    p.add_argument('--profile_root', default=None)
    p.add_argument('--no_seed', action='store_true')
    p.add_argument('--max_evals', type=int, default=None)
    p.add_argument('--max_evals_per_layer', type=int, default=None)
    p.add_argument('--search_repeats', type=int, default=1)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--radius', type=int, default=None)
    p.add_argument('--time_budget_s', type=float, default=None)
    p.add_argument('--layers', default=None)
    p.add_argument('--assignment_out', default='output/window_search/assignment.yaml')
    p.add_argument('--trace_out', default='output/window_search/search_trace.csv')
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


def parse_layers(spec, n_layers):
    if not spec:
        return None

    layers = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if '-' in chunk.lstrip('-'):
            lo, hi = chunk.split('-', 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(chunk))

    unknown = [l for l in layers if not 0 <= l < n_layers]
    if unknown:
        raise ValueError(f"--layers {spec!r} names layers {unknown}, but the model has "
                         f"{n_layers} (0..{n_layers - 1})")
    return layers


def resolve_noise_floor(args):
    if args.noise_floor is not None:
        return args.noise_floor, f"--noise_floor {args.noise_floor:g}"

    if args.noise_summary:
        summary = load_yaml(args.noise_summary)
        spread = summary.get('ppl_spread')
        if spread is None:
            raise ValueError(f"{args.noise_summary} has no 'ppl_spread'; it does not look "
                             f"like a summary written by --mode noise")
        return float(spread), f"ppl_spread from {args.noise_summary}"

    raise SystemExit("--mode search needs a noise floor: pass --noise_floor, or "
                     "--noise_summary pointing at the *_summary.yaml that --mode noise "
                     "wrote. A search run with an unmeasured floor cannot tell an "
                     "improvement from run-to-run noise.")


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


def bootstrap_ci(batch_losses, n_resamples=10000, confidence=95.0, seed=0):
    losses = [x for x in batch_losses if x == x]
    n = len(losses)
    if n < 2:
        return {'bootstrap_n_batches': n, 'bootstrap_resamples': 0,
                'ppl_ci_low': float('nan'), 'ppl_ci_high': float('nan'),
                'ppl_ci_width': float('nan')}

    rng = random.Random(seed)
    draws = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += losses[rng.randrange(n)]
        draws.append(math.exp(total / n))
    draws.sort()

    tail = (100.0 - confidence) / 200.0
    lo = draws[int(tail * n_resamples)]
    hi = draws[min(n_resamples - 1, int((1.0 - tail) * n_resamples))]
    return {'bootstrap_n_batches': n, 'bootstrap_resamples': n_resamples,
            'ppl_ci_low': lo, 'ppl_ci_high': hi, 'ppl_ci_width': hi - lo}


def reload_savings(load_s, eval_s, n_evals, n_layers):
    search_s = n_evals * eval_s
    reloading_s = n_layers * load_s + search_s
    resident_s = load_s + search_s
    return {
        'n_layers': n_layers,
        'n_evals': n_evals,
        'load_s': load_s,
        'eval_s_median': eval_s,
        'reloading_s': reloading_s,
        'resident_s': resident_s,
        'saved_s': (n_layers - 1) * load_s,
        'speedup': reloading_s / resident_s if resident_s else float('nan'),
    }


def print_savings(savings):
    print()
    print("reload cost (Dr. Wu 8/03: 'every time loading versus no loading')")
    print(f"  model load           {savings['load_s']:.1f}s")
    print(f"  eval (median)        {savings['eval_s_median']:.2f}s")
    print(f"  reload per layer     {savings['n_layers']} x load + "
          f"{savings['n_evals']} x eval = {savings['reloading_s'] / 60:.1f} min")
    print(f"  model resident       1 x load + "
          f"{savings['n_evals']} x eval = {savings['resident_s'] / 60:.1f} min")
    print(f"  saved                {savings['saved_s'] / 60:.1f} min "
          f"({savings['speedup']:.1f}x)")
    print("  lower bound: charges nothing for Slurm queue wait or for editing the "
          "YAML by hand between jobs")


def run_noise(args, harness, load_timing):
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
    stats.update(bootstrap_ci(results[0].batch_losses))
    path = harness.write_csv(args.out)

    print()
    print(f"ppl mean          {stats['ppl_mean']:.6f}")
    print(f"ppl spread        {stats['ppl_spread']:.6g}")
    print(f"ppl stdev         {stats['ppl_stdev']:.6g}")
    print(f"ppl ci95          [{stats['ppl_ci_low']:.6f}, {stats['ppl_ci_high']:.6f}] "
          f"width {stats['ppl_ci_width']:.6g} "
          f"over {stats['bootstrap_n_batches']} batches")
    print(f"wall/eval median  {stats['wall_s_median']:.2f}s")
    print(f"apply/eval median {stats['apply_ms_median']:.2f}ms")
    print(f"tokens/eval       {stats['n_tokens']}")
    print(f"wrote             {os.path.abspath(path)}")

    savings = reload_savings(load_timing.load_s, stats['wall_s_median'],
                             n_evals=16 * harness.n_layers, n_layers=harness.n_layers)
    print_savings(savings)

    summary_path = os.path.splitext(path)[0] + '_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.safe_dump({**stats,
                        'baseline_ppl': harness.baseline_ppl,
                        'assignment_digest': assignment.digest(),
                        'input_hash': harness.input_hash,
                        'load': load_timing.to_row(),
                        'projected_savings': savings,
                        'provenance': results[0].provenance}, f, sort_keys=True)
    print(f"wrote             {os.path.abspath(summary_path)}")
    return 0


def run_search(args, harness, load_timing):
    noise_floor, source = args._noise_floor
    print(f"noise floor: {noise_floor:.6g} ({source})")

    if noise_floor >= 0.007:
        print(f"warning: the per-layer gain reported in the paper is 0.23 PPL over 32 "
              f"layers, about 0.007 per layer. A noise floor of {noise_floor:.6g} is at "
              f"or above that, so most layers will correctly refuse to move. Raise "
              f"--search_repeats until the floor drops below it.", file=sys.stderr)

    seeder = None
    if args.profile_root and not args.no_seed:
        seeder = HistogramSeeder(args.profile_root, exp_dim=args.exp_dim,
                                 group_size=args.group_size)
        if seeder.exp_dist_root is None:
            print(f"warning: no exponent histograms under {args.profile_root!r}; every "
                  f"layer will fall back to the default seed", file=sys.stderr)
        else:
            print(f"seeding from {seeder.exp_dist_root} "
                  f"({len(seeder.available_layers())} layers)")
    elif args.no_seed:
        print("seeding disabled (--no_seed): every layer starts from "
              f"{harness.seed_attention}")

    budget = SearchBudget(max_evals=args.max_evals,
                          max_evals_per_layer=args.max_evals_per_layer,
                          repeats=args.search_repeats, noise_floor=noise_floor,
                          patience=args.patience, time_budget_s=args.time_budget_s)

    search = ProgressiveLayerSearch(
        harness, seeder=seeder, budget=budget,
        grid=CandidateGrid(radius=args.radius),
        layers=parse_layers(args.layers, harness.n_layers))

    result = search.run()
    result.load_s = load_timing.load_s

    print()
    if result.baseline_ppl is not None:
        print(f"baseline (unpatched)  {result.baseline_ppl:.6f}")
    print(f"seed assignment       {result.seed_ppl:.6f}")
    print(f"per-layer search      {result.final_ppl:.6f}")
    print(f"improvement           {result.improvement:.6f} over "
          f"{result.n_layers_changed}/{len(result.outcomes)} layers")
    print(f"evaluations           {result.n_evals}")
    print(f"wall                  {result.wall_s / 60:.1f} min")

    print_savings(result.savings_report(harness.n_layers))

    trace = result.write_csv(args.trace_out)
    assignment_path = result.assignment.to_yaml(args.assignment_out)
    summary_path = os.path.splitext(args.trace_out)[0] + '_summary.yaml'
    result.write_yaml(summary_path)
    harness.write_csv(args.out)

    print()
    print(f"wrote             {os.path.abspath(trace)}")
    print(f"wrote             {os.path.abspath(assignment_path)}")
    print(f"wrote             {os.path.abspath(summary_path)}")
    print(f"wrote             {os.path.abspath(args.out)}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)

    if args.mode == 'search':
        args._noise_floor = resolve_noise_floor(args)
        if args.profile_root and not args.no_seed and not os.path.isdir(args.profile_root):
            raise SystemExit(f"--profile_root {args.profile_root!r} does not exist")

    loader = ModelLoader()
    with timed_load(cfg.model_id) as timing:
        model = loader.load(cfg)
    load_timing = timing[0].with_stages(loader.timings).with_model(model)

    params = (f"{load_timing.n_params / 1e9:.2f}B params"
              if load_timing.n_params else "unknown size")
    print(f"model load: {load_timing.load_s:.1f}s total, "
          f"{load_timing.weights_s:.1f}s weights, {params} on {load_timing.device}")

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

    if args.mode == 'search':
        return run_search(args, harness, load_timing)
    return run_noise(args, harness, load_timing)


if __name__ == '__main__':
    sys.exit(main())

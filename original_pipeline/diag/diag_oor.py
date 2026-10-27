import argparse
import csv
import math
import os
import sys
import tempfile
from collections import defaultdict

import torch

MASK_THRESHOLD = -65500.0
SEED_PPL_8DOC = 3.9108295166396467
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SM_CATS = ('sm_above_lut', 'sm_below_slide', 'sm_below_lut')
FFN_LUT_CATS = ('ffn_pos_above', 'ffn_pos_below', 'ffn_neg_above', 'ffn_neg_below')
FFN_CATS = FFN_LUT_CATS + ('ffn_below_slide',)

MODES = {
    'as_is': {},
    'exact_sm_above': {'sm_above_lut': 'exact'},
    'exact_sm_below': {'sm_below_slide': 'exact', 'sm_below_lut': 'exact'},
    'exact_ffn_lut': {c: 'exact' for c in FFN_LUT_CATS},
    'exact_ffn_slide': {'ffn_below_slide': 'exact'},
    'exact_all': {c: 'exact' for c in SM_CATS + FFN_CATS},
    'alt_sm_below_one': {'sm_below_slide': 'one', 'sm_below_lut': 'one'},
    'alt_ffn_below_half': {'ffn_pos_below': 'half', 'ffn_neg_below': 'half'},
    'alt_ffn_slide_zero': {'ffn_below_slide': 'zero'},
}

FIELDS = ['exp_dim', 'mode', 'ppl', 'gain_vs_as_is', 'paired_ci_low', 'paired_ci_high',
          'num_batches', 'n_tokens', 'input_hash', 'wall_s', 'batch_losses']
STAT_FIELDS = ['exp_dim', 'layer', 'site', 'category', 'count', 'n_unmasked', 'fraction',
               'exact_mass_per_row']

STATE = {'mode': {}, 'stats': False, 'counts': None, 'mass': None, 'rows': None}


def softmax_categories(self, raw, window_exp, max_exp_mask, masked, zero):
    live = ~masked & ~zero
    above = max_exp_mask & live
    clamped_up = live & ~above & (raw < window_exp)
    return {
        'sm_above_lut': above,
        'sm_below_slide': clamped_up & (raw >= self.min_exp),
        'sm_below_lut': clamped_up & (raw < self.min_exp),
        'sm_above_slide_in_lut': live & ~above & (raw > window_exp),
    }


def softmax_nonlinear(self, attn_weights, dim=-1, dtype=torch.bfloat16):
    if self.lut_build != 'max':
        raise SystemExit('diag_oor categories assume max-built softmax windows')
    attn_weights = attn_weights.to(torch.bfloat16)
    attn_weights_max = torch.max(attn_weights, dim=dim, keepdim=True)[0]
    attn_weights = attn_weights - attn_weights_max
    del attn_weights_max

    mant, exp = torch.frexp(attn_weights)
    mant.mul_(16).round_()
    exp[attn_weights != 0] -= 1
    mant = mant.to(torch.int32)
    mant.abs_()
    exp[mant == 16] += 1

    max_exp_mask = exp > self.max_exp
    mant &= 0x7
    masked = attn_weights <= MASK_THRESHOLD
    exp[masked] = 0
    mode = STATE['mode']
    tagging = bool(mode) or STATE['stats']
    raw = exp.clone() if tagging else None

    exp, mant = self.window_softmax_approx(exp, mant)
    window_exp = exp + self.min_exp if tagging else None
    exponentials = self.lut[exp, mant]
    del mant, exp

    zero = attn_weights == 0
    exponentials[zero] = 1
    exponentials[max_exp_mask] = 0

    if tagging:
        cats = softmax_categories(self, raw, window_exp, max_exp_mask, masked, zero)
        del raw, window_exp
        if STATE['stats']:
            record_softmax(self.layer, cats, attn_weights, masked, dim)
        for cat, rule in mode.items():
            if cat not in cats:
                continue
            sel = cats[cat]
            if rule == 'exact':
                exponentials[sel] = torch.exp(attn_weights[sel].float()).to(exponentials.dtype)
            elif rule == 'one':
                exponentials[sel] = 1
        del cats
    del zero

    exponentials[masked] = 0
    del max_exp_mask, masked

    attn_weights = torch.sum(exponentials, dim=dim, keepdim=True)
    attn_weights = exponentials / attn_weights
    del exponentials
    return attn_weights


def silu_categories(self, x, exp, pos_exp, neg_exp):
    positive = x > 0
    negative = x < 0
    pos_in = positive & (exp >= self.pos_min_exp) & (exp <= self.max_pos_exp)
    neg_in = negative & (exp >= self.neg_min_exp) & (exp <= self.max_neg_exp)
    pos_window = pos_exp + self.pos_min_exp
    neg_window = neg_exp + self.neg_min_exp
    return {
        'ffn_pos_above': positive & (exp > self.max_pos_exp),
        'ffn_pos_below': positive & (exp < self.pos_min_exp),
        'ffn_neg_above': negative & (exp > self.max_neg_exp),
        'ffn_neg_below': negative & (exp < self.neg_min_exp),
        'ffn_below_slide': (pos_in & (pos_window > exp)) | (neg_in & (neg_window > exp)),
        'ffn_above_slide_in_lut': (pos_in & (pos_window < exp)) | (neg_in & (neg_window < exp)),
    }


def silu_nonlinear(self, x):
    x = x.to(torch.bfloat16)
    mant, exp = torch.frexp(x)
    mant = torch.round(mant * 16)
    exp = torch.where(torch.abs(mant) == 16, exp + 1, exp)
    exp = torch.where(x == 0, exp, exp - 1)
    mant = torch.abs(mant.to(torch.int32)) & 0x7
    exp = exp.to(torch.int32)
    mant = mant.to(torch.int32)

    pos_exp, pos_mant, neg_exp, neg_mant = self.window_silu_approx(exp, mant)
    pos_mant = pos_mant.to(torch.int32)
    neg_mant = neg_mant.to(torch.int32)
    silu = self.lut_index(x, pos_exp, pos_mant, neg_exp, neg_mant, exp)

    mode = STATE['mode']
    if mode or STATE['stats']:
        cats = silu_categories(self, x, exp, pos_exp, neg_exp)
        if STATE['stats']:
            record_silu(self.layer, cats, x)
        for cat, rule in mode.items():
            if cat not in cats:
                continue
            sel = cats[cat]
            if rule == 'exact':
                silu[sel] = torch.nn.functional.silu(x[sel].float()).to(silu.dtype)
            elif rule == 'half':
                silu[sel] = (x[sel].float() / 2).to(silu.dtype)
            elif rule == 'zero':
                silu[sel] = 0
        del cats
    return silu


def record_softmax(layer, cats, attn_weights, masked, dim):
    unmasked = int((~masked).sum().item())
    probs = torch.exp(attn_weights.float()).masked_fill(masked, 0.0)
    probs = probs / probs.sum(dim=dim, keepdim=True)
    rows = probs.numel() // probs.shape[dim]
    for cat, sel in cats.items():
        key = (layer, 'attention', cat)
        STATE['counts'][key][0] += int(sel.sum().item())
        STATE['counts'][key][1] += unmasked
        STATE['mass'][key] += float(probs[sel].sum().item())
        STATE['rows'][key] += rows
    del probs


def record_silu(layer, cats, x):
    total = x.numel()
    for cat, sel in cats.items():
        key = (layer, 'ffn', cat)
        STATE['counts'][key][0] += int(sel.sum().item())
        STATE['counts'][key][1] += total


def install():
    from custom_nonlinear.custom_nonlinear_functions.vlp_silu_approx import VLPSilu
    from custom_nonlinear.custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax
    originals = {'softmax': VLPSoftmax.nonlinear, 'silu': VLPSilu.nonlinear}
    VLPSoftmax.nonlinear = softmax_nonlinear
    VLPSilu.nonlinear = silu_nonlinear
    return VLPSoftmax, VLPSilu, originals


def self_check():
    from profiling_api.windows import FfnWindow, SoftmaxWindow
    VLPSoftmax, VLPSilu, originals = install()
    common = dict(layer=0, device='cpu', profile_dims=[0], keys=[], profile_path=tempfile.mkdtemp() + '/')
    torch.manual_seed(0)
    q = 256
    scores = torch.randn(1, 2, q, q) * 6.0 - 10.0
    near = (torch.arange(q) % 4 == 0).view(1, 1, q)
    scores[..., 0] = 0.0
    scores[..., 1] = torch.where(near, torch.tensor(-0.02), torch.tensor(-20.0))
    scores[..., 2] = torch.where(near, torch.tensor(-0.003), torch.tensor(-20.0))
    scores = scores.clamp(max=0.0).to(torch.float16)
    mask = torch.triu(torch.full((q, q), torch.finfo(torch.float16).min, dtype=torch.float16), 1)
    scores = scores + mask
    acts = torch.randn(4, 3, 11008) * 1.5
    acts[..., ::97] *= 40.0
    acts[..., 1::13] *= 0.004
    acts = acts.to(torch.bfloat16)
    failures = []
    for exp_dim in (10, 6):
        sm = VLPSoftmax(**SoftmaxWindow(exp_dim=exp_dim, anchor=2).to_kwargs(), **common)
        fn = VLPSilu(**FfnWindow(exp_dim=exp_dim, pos_anchor=2, neg_anchor=2).to_kwargs(), **common)
        ref_sm = originals['softmax'](sm, scores.clone())
        ref_fn = originals['silu'](fn, acts.clone())
        future = torch.triu(torch.ones(q, q, dtype=torch.bool), 1).expand_as(ref_sm)
        for name, rules in MODES.items():
            STATE['mode'] = rules
            out_sm = sm.nonlinear(scores.clone())
            out_fn = fn.nonlinear(acts.clone())
            if not torch.all(out_sm[future] == 0):
                failures.append(f'{exp_dim}/{name}: masked entries got weight')
            if not torch.isfinite(out_sm).all() or not torch.isfinite(out_fn).all():
                failures.append(f'{exp_dim}/{name}: non-finite output')
            if name == 'as_is':
                if not torch.equal(out_sm, ref_sm):
                    failures.append(f'{exp_dim}: softmax wrapper differs from the kernel')
                if not torch.equal(out_fn, ref_fn):
                    failures.append(f'{exp_dim}: SiLU wrapper differs from the kernel')
        STATE['mode'] = {}
        STATE['stats'] = True
        STATE['counts'] = defaultdict(lambda: [0, 0])
        STATE['mass'] = defaultdict(float)
        STATE['rows'] = defaultdict(int)
        sm.nonlinear(scores.clone())
        fn.nonlinear(acts.clone())
        STATE['stats'] = False
        seen = {cat: c[0] for (_, _, cat), c in STATE['counts'].items()}
        for cat in ('sm_above_slide_in_lut', 'ffn_above_slide_in_lut'):
            if seen.get(cat, 0):
                failures.append(f'{exp_dim}: {cat} = {seen[cat]}, expected 0 for max-built windows')
        print(f'exp_dim {exp_dim} category counts on synthetic tensors: {seen}')
        STATE['mode'] = MODES['exact_all']
        out_fn = fn.nonlinear(acts.clone())
        cats = silu_categories(fn, acts, *frexp_parts(fn, acts))
        oor = torch.zeros_like(acts, dtype=torch.bool)
        for cat in FFN_CATS:
            oor |= cats[cat]
        exact = torch.nn.functional.silu(acts.float()).to(out_fn.dtype)
        if not torch.equal(out_fn[oor], exact[oor]):
            failures.append(f'{exp_dim}: exact_all SiLU differs from silu on out-of-range elements')
        STATE['mode'] = {}
    if failures:
        print('SELF-CHECK FAILED:\n  ' + '\n  '.join(failures))
        return 1
    print('self-check passed: as_is wrappers are bit-identical to the kernels; masked entries stay 0 '
          'in every mode; exact_all matches silu on out-of-range elements')
    return 0


def frexp_parts(fn, x):
    mant, exp = torch.frexp(x)
    mant = torch.round(mant * 16)
    exp = torch.where(torch.abs(mant) == 16, exp + 1, exp)
    exp = torch.where(x == 0, exp, exp - 1)
    mant = torch.abs(mant.to(torch.int32)) & 0x7
    exp = exp.to(torch.int32)
    pos_exp, _, neg_exp, _ = fn.window_silu_approx(exp, mant.to(torch.int32))
    return exp, pos_exp, neg_exp


def ppl_of(losses):
    total = 0.0
    for value in losses:
        total += value
    return math.exp(total / len(losses))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=REPO)
    p.add_argument('--n_samples', type=int, default=16)
    p.add_argument('--exp_dims', default='10,6')
    p.add_argument('--modes', default=','.join(MODES))
    p.add_argument('--out', default=None)
    p.add_argument('--stats_out', default=None)
    p.add_argument('--self_check', action='store_true')
    args = p.parse_args(argv)

    out = os.path.abspath(args.out) if args.out else None
    stats_out = os.path.abspath(args.stats_out) if args.stats_out else None
    os.chdir(args.repo)
    sys.path.insert(0, os.getcwd())
    if args.self_check:
        return self_check()
    if not (out and stats_out):
        raise SystemExit('--out and --stats_out are required for a GPU run')

    from profiling_api.evaluate import WindowEvalHarness
    from profiling_api.pipeline import ModelLoader
    from profiling_api.search import paired_ppl_ci
    from profiling_api.windows import FfnWindow, SoftmaxWindow
    from run_window_search import build_config, parse_args

    install()
    cfg = build_config(parse_args([
        '--model_config', 'config/model_config/llama/llama_2_7b.yaml',
        '--nonlinear_config', 'config/nonlinear_config/nonlinear_config.yaml',
        '--parameter_config', 'config/parameter_config/parameter_config.yaml',
        '--n_samples', str(args.n_samples)]))
    print('Loading model', flush=True)
    model = ModelLoader().load(cfg)
    model.model.eval()
    harness = WindowEvalHarness(model, seed_attention=SoftmaxWindow(exp_dim=10, anchor=2),
                                seed_ffn=FfnWindow(exp_dim=10, pos_anchor=2, neg_anchor=2))
    harness.setup(measure_baseline=True)
    print(f'baseline (unpatched) ppl {harness.baseline_ppl!r} over {args.n_samples} docs, '
          f'input_hash {harness.input_hash}', flush=True)

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown or modes[0] != 'as_is':
        raise SystemExit(f'--modes must start with as_is and use known modes; unknown {unknown}')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(os.path.dirname(stats_out), exist_ok=True)
    print('Looping through configurations', flush=True)
    stat_rows = []
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({'exp_dim': '', 'mode': 'unpatched', 'ppl': harness.baseline_ppl,
                         'num_batches': harness.baseline_result.num_batches,
                         'n_tokens': harness.n_tokens, 'input_hash': harness.input_hash,
                         'wall_s': round(harness.baseline_result.wall_s, 2),
                         'batch_losses': ' '.join(repr(v) for v in harness.baseline_result.batch_losses)})
        for exp_dim in [int(d) for d in args.exp_dims.split(',')]:
            assignment = harness.uniform(SoftmaxWindow(exp_dim=exp_dim, anchor=2),
                                         FfnWindow(exp_dim=exp_dim, pos_anchor=2, neg_anchor=2))
            reference = None
            for mode in modes:
                STATE['mode'] = MODES[mode]
                STATE['stats'] = mode == 'as_is'
                if STATE['stats']:
                    STATE['counts'] = defaultdict(lambda: [0, 0])
                    STATE['mass'] = defaultdict(float)
                    STATE['rows'] = defaultdict(int)
                result = harness.evaluate(assignment)
                STATE['mode'] = {}
                losses = list(result.batch_losses)
                row = {'exp_dim': exp_dim, 'mode': mode, 'ppl': result.ppl,
                       'num_batches': result.num_batches, 'n_tokens': result.n_tokens,
                       'input_hash': result.input_hash, 'wall_s': round(result.wall_s, 2),
                       'batch_losses': ' '.join(repr(v) for v in losses)}
                if mode == 'as_is':
                    reference = losses
                    if exp_dim == 10 and len(losses) >= 4:
                        first8 = ppl_of(losses[:4])
                        print(f'reproduction check: exp(mean of first 4 batch losses) = {first8!r}, '
                              f'committed 8-doc seed = {SEED_PPL_8DOC!r}, '
                              f'delta {first8 - SEED_PPL_8DOC:.3g}', flush=True)
                        if abs(first8 - SEED_PPL_8DOC) > 1e-9:
                            writer.writerow(row)
                            raise SystemExit('reproduction check FAILED: the as_is wrapper or the '
                                             'documents differ from the committed seed run; no other '
                                             'number from this job can be trusted')
                    for (layer, site, cat), (count, n) in sorted(STATE['counts'].items()):
                        key = (layer, site, cat)
                        stat_rows.append({
                            'exp_dim': exp_dim, 'layer': layer, 'site': site, 'category': cat,
                            'count': count, 'n_unmasked': n, 'fraction': count / n if n else '',
                            'exact_mass_per_row': (STATE['mass'][key] / STATE['rows'][key]
                                                   if STATE['rows'].get(key) else '')})
                    STATE['stats'] = False
                else:
                    ci = paired_ppl_ci(reference, losses)
                    row['gain_vs_as_is'] = ppl_of(reference) - result.ppl
                    if ci is not None:
                        row['paired_ci_low'], row['paired_ci_high'] = ci
                writer.writerow(row)
                f.flush()
                print({k: row.get(k, '') for k in FIELDS if k != 'batch_losses'}, flush=True)

    with open(stats_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=STAT_FIELDS)
        writer.writeheader()
        writer.writerows(stat_rows)
    print(f'Saving results to {out} and {stats_out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

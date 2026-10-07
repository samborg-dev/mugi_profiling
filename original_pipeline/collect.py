import os
import statistics
import sys

import pandas as pd
import yaml

HOME = os.path.expanduser('~')
WORK = os.path.join(HOME, 'orig_compare')
AE = os.path.join(HOME, 'mugi_ae', 'csv', 'weights', 'llama-2-7b-hf')
NEWTON = os.path.join(HOME, 'mugi_newton')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ae_merge import EXPECT
import newton_next

PARAM_COLS = ['attn_exp_dim', 'attn_max_exp', 'attn_min_exp', 'attn_lut_build',
              'attn_segments', 'attn_segment_0', 'attn_degree_center', 'attn_degrees',
              'ffn_exp_dim', 'ffn_max_pos_exp', 'ffn_min_pos_exp', 'ffn_lut_build',
              'ffn_segments', 'ffn_segment_0']


def params_of(row):
    return {c: (row[c].item() if hasattr(row[c], 'item') else row[c])
            for c in PARAM_COLS if c in row and pd.notna(row[c])}


def ae_perplexity():
    out = {}
    splits = [os.path.join(AE, f'nonlinear_config_split_{s}', 'metric.csv') for s in EXPECT]
    frames = [pd.read_csv(p) for p in splits if os.path.exists(p)]
    if not frames:
        return out
    df = pd.concat(frames, ignore_index=True)
    base = df[df['function_name'] == 'torch']
    if len(base):
        out['baseline'] = float(base['value'].iloc[0])
    for (fn, pa, pf), group in df[df['function_name'] != 'torch'].groupby(
            ['function_name', 'patch_attention', 'patch_ffn']):
        site = 'softmax' if pa else 'silu'
        best = group.loc[group['value'].idxmin()]
        out[f'best uniform {fn} {site}'] = {'ppl': float(best['value']), 'evals': len(group),
                                            'params': params_of(best)}
    e2e = os.path.join(AE, 'nonlinear_e2e_config', 'metric.csv')
    if os.path.exists(e2e):
        for _, row in pd.read_csv(e2e).iterrows():
            out[f"end to end {row['function_name']}"] = float(row['value'])
    return out


def newton_perplexity():
    root = os.path.join(NEWTON, newton_next.CSV_ROOT)
    rows = []
    for layer in range(1, newton_next.N_LAYERS + 1):
        if newton_next.candidates(layer, root) is None:
            break
        anchor, side, ppl = newton_next.pick(layer, root)
        rows.append({'layer': layer, 'side': side, 'anchor': anchor, 'ppl': ppl})
    if not rows:
        return {}, rows
    first = newton_next.candidates(1, root)
    uniform = first[(first['lut_build'] == 'max') & (first['anchor'] == 2)]
    return {'uniform start (all layers max+2)': float(uniform['value'].iloc[0]),
            'per-layer final': rows[-1]['ppl'],
            'layers done': len(rows)}, rows


def unit_evals(pipeline, unit):
    if pipeline == 'newton':
        return len(newton_next.MAX_VALUES) + len(newton_next.MIN_VALUES)
    if unit == 'e2e':
        return 3
    split, chunk = unit.split('_')[1], unit.split('_')[2]
    return EXPECT[split][chunk]


def efficiency():
    path = os.path.join(WORK, 'unit_times.csv')
    if not os.path.exists(path):
        return {}
    times = pd.read_csv(path)
    ok = times[times['rc'] == 0].drop_duplicates(subset=['pipeline', 'unit'], keep='last')
    out = {}
    for pipeline, group in ok.groupby('pipeline'):
        loads, per_eval, walls, evals, gpu_s = [], [], [], 0, 0.0
        for _, r in group.iterrows():
            n = unit_evals(pipeline, r['unit'])
            wall = r['end'] - r['start']
            walls.append(wall)
            evals += n
            gpus = r['gpus'] if pd.notna(r['gpus']) and r['gpus'] else 2
            gpu_s += wall * float(gpus)
            if pd.notna(r['loop_start']):
                loads.append(r['loop_start'] - r['start'])
                per_eval.append((r['end'] - r['loop_start']) / n)
        out[pipeline] = {
            'units': len(group),
            'evals': evals,
            'model_loads': len(group),
            'load_s_median': statistics.median(loads) if loads else None,
            'eval_s_median': statistics.median(per_eval) if per_eval else None,
            'wall_min_total': sum(walls) / 60,
            'gpu_hours': gpu_s / 3600,
            'failed_attempts': int((times[times['pipeline'] == pipeline]['rc'] != 0).sum()),
        }
    return out


def unpatched_warnings(ae):
    base = ae.get('baseline')
    if base is None:
        return []
    out = []
    for key, value in ae.items():
        ppl = value['ppl'] if isinstance(value, dict) else value
        if key != 'baseline' and ppl == base:
            out.append(f'{key} equals the unpatched baseline exactly; its approximation was not applied')
    return out


def main():
    ae = ae_perplexity()
    nw, layers = newton_perplexity()
    eff = efficiency()
    report = {'warnings': unpatched_warnings(ae), 'artifact': ae, 'newton': nw,
              'newton_layers': layers, 'efficiency': eff}
    with open(os.path.join(WORK, 'summary.yaml'), 'w') as f:
        yaml.safe_dump(report, f, sort_keys=False)
    print(yaml.safe_dump(report, sort_keys=False, default_flow_style=None))
    print(f"wrote {os.path.join(WORK, 'summary.yaml')}")


if __name__ == '__main__':
    main()

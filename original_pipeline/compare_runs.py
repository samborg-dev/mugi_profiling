import argparse
import os
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newton_next
from ae_merge import EXPECT
from collect import unpatched_warnings

N_LAYERS = 32
GPUS = 2

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
NEWEST = '#2a78d6'
NEWEST_FIXED = '#8a4fd1'
NEWTON = '#eb6834'
ARTIFACT = '#1baf7a'
COLORS = {'newest': NEWEST, 'newest_fixed': NEWEST_FIXED, 'newton': NEWTON, 'artifact': ARTIFACT}
SEARCH_RUNS = ('newest', 'newest_fixed', 'newton')
LABELS = {
    'newest': 'Newest edition, original kernel (mask leak)',
    'newest_fixed': 'Newest edition, fixed kernel',
    'newton': 'newton edition (manual per-layer), unfixed kernel',
    'artifact': 'Paper artifact (uniform grid), unfixed kernel',
}
DISPLAY = {
    'newest': 'Newest, unfixed kernel',
    'newest_fixed': 'Newest, fixed kernel',
    'newton': 'newton (manual)',
    'artifact': 'Paper artifact (uniform)',
}
PPL_AXIS = 'PPL (lower = better)'
LAYER_AXIS = 'Layers tuned'
HIGHLIGHT = '#eb6834'
RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SIDE_FILL = {'max': '#f0efec', 'min': '#c3c2b7'}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans'],
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': BASELINE,
    'axes.labelcolor': INK2,
    'text.color': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.titlelocation': 'left',
    'axes.labelsize': 11,
    'legend.frameon': False,
    'savefig.facecolor': SURFACE,
    'savefig.dpi': 200,
})

LUT_SIZE = 'LUT size (exponents stored)'
HEATMAPS = [
    ('VLP softmax, max-built', dict(function_name='vlp', patch_attention=True, patch_ffn=False,
                                    attn_lut_build='max'), 'attn_exp_dim', 'attn_max_exp',
     LUT_SIZE, 'LUT max exponent'),
    ('VLP softmax, min-built', dict(function_name='vlp', patch_attention=True, patch_ffn=False,
                                    attn_lut_build='min'), 'attn_exp_dim', 'attn_min_exp',
     LUT_SIZE, 'LUT min exponent'),
    ('VLP SiLU, max-built', dict(function_name='vlp', patch_attention=False, patch_ffn=True,
                                 ffn_lut_build='max'), 'ffn_exp_dim', 'ffn_max_pos_exp',
     LUT_SIZE, 'LUT max exponent'),
    ('VLP SiLU, min-built', dict(function_name='vlp', patch_attention=False, patch_ffn=True,
                                 ffn_lut_build='min'), 'ffn_exp_dim', 'ffn_min_pos_exp',
     LUT_SIZE, 'LUT min exponent'),
    ('PWL softmax', dict(function_name='pwl', patch_attention=True, patch_ffn=False),
     'attn_segments', 'attn_segment_0', 'Segments', 'Segment range sr (covers [sr, 0])'),
    ('PWL SiLU', dict(function_name='pwl', patch_attention=False, patch_ffn=True),
     'ffn_segments', 'ffn_segment_0', 'Segments', 'Segment range sr (covers [−sr, sr])'),
    ('Taylor softmax', dict(function_name='taylor', patch_attention=True, patch_ffn=False),
     'attn_degrees', 'attn_degree_center', 'Degrees (polynomial terms)', 'Expansion center'),
]

PAPER_E2E_100DOC = {'Base': 5.75, 'VLP': 6.21, 'PWL': 5.68, 'T': 5.78}


def check(condition, message):
    if not condition:
        raise SystemExit(f'check failed: {message}')


def latest_success(times, pipeline):
    rows = times[(times['pipeline'] == pipeline) & (times['rc'] == 0)]
    return rows.drop_duplicates(subset=['unit'], keep='last').sort_values('unit')


def load_newest(root, tag, pipeline='newest', expect_evals=None):
    base = os.path.join(root, 'output', 'window_search')
    with open(os.path.join(base, f'search_trace_{tag}_summary.yaml')) as f:
        data = yaml.safe_load(f)
    trace = pd.read_csv(os.path.join(base, f'search_trace_{tag}.csv'))
    cost = pd.read_csv(os.path.join(base, f'cost_and_noise_{tag}.csv'))
    summary = data['summary']
    curve = pd.DataFrame(data['locking_curve'])
    layers = pd.DataFrame(data['layers']).sort_values('layer')

    if expect_evals is not None:
        check(int(summary['n_evals']) == expect_evals,
              f"{pipeline} n_evals {summary['n_evals']} != {expect_evals}")
    check(len(layers) == N_LAYERS, f'{pipeline} has {len(layers)} layer rows')

    load_s = float(summary['load_s'])
    baseline_wall = float(cost.loc[cost['assignment_hash'] == 'baseline-unpatched', 'wall_s'].iloc[0])
    seed_wall = float(cost.loc[cost['assignment_hash'] != 'baseline-unpatched', 'wall_s'].iloc[0])
    lead_s = load_s + baseline_wall + seed_wall
    gpus = int(cost['prov_n_devices'].iloc[0]) if 'prov_n_devices' in cost.columns else GPUS

    accepted = trace[(~trace['is_seed']) & (trace['accepted'])].groupby('layer').size()
    rows = [{'pipeline': pipeline, 'layer': 0, 'ppl': float(summary['seed_ppl']), 'step': 0.0,
             'cumulative_evals': 1, 'cumulative_compute_s': lead_s, 'side': None,
             'anchor': None, 'accepted': False, 'stop_reason': None, 'searched': True}]
    by_layer = layers.set_index('layer')
    for _, r in curve[curve['n_locked'] > 0].iterrows():
        layer = int(r['layer'])
        info = by_layer.loc[layer]
        rows.append({'pipeline': pipeline, 'layer': layer + 1, 'ppl': float(r['ppl']),
                     'step': float(r['step']), 'cumulative_evals': 1 + int(r['cumulative_evals']),
                     'cumulative_compute_s': lead_s + float(r['cumulative_wall_s']),
                     'side': info['chosen_anchor_side'], 'anchor': int(info['chosen_anchor']),
                     'accepted': bool(accepted.get(layer, 0)),
                     'stop_reason': info['stop_reason'], 'searched': int(info['n_evals']) > 0})
    curve_df = pd.DataFrame(rows)

    run = {
        'pipeline': pipeline, 'label': LABELS[pipeline], 'gpus': gpus,
        'coverage': 'softmax + SiLU', 'baseline': float(summary['baseline_ppl']),
        'start_ppl': float(summary['seed_ppl']), 'final_ppl': float(summary['final_ppl']),
        'evals': int(summary['n_evals']), 'loads_as_designed': 1, 'loads_as_run': 1,
        'load_s_median': load_s, 'eval_s_median': float(summary['eval_s_median']),
        'compute_s': load_s + baseline_wall + float(summary['wall_s']),
        'load_s_total': load_s, 'eval_s_total': baseline_wall + float(summary['wall_s']),
        'manual_steps': 0, 'candidates_per_layer': 3,
    }
    return run, curve_df, trace


def load_newton(lab, times):
    root = os.path.join(lab, 'newton_csv')
    rows = []
    start = newton_next.candidates(1, root)
    check(start is not None, 'newton layer 1 missing')
    uniform = float(start[(start['lut_build'] == 'max') & (start['anchor'] == 2)]['value'].iloc[0])

    units = latest_success(times, 'newton').set_index('unit')
    check(len(units) == N_LAYERS, f'newton has {len(units)} timed layers, expected {N_LAYERS}')
    per_layer = 16
    cumulative_s = 0.0
    previous = uniform
    rows.append({'pipeline': 'newton', 'layer': 0, 'ppl': uniform, 'step': 0.0,
                 'cumulative_evals': 0, 'cumulative_compute_s': 0.0, 'side': None,
                 'anchor': None, 'accepted': False, 'stop_reason': None, 'searched': True})
    loads, evals_s = [], []
    for layer in range(1, N_LAYERS + 1):
        df = newton_next.candidates(layer, root)
        check(df is not None and len(df) == per_layer, f'newton layer {layer} incomplete')
        anchor, side, ppl = newton_next.pick(layer, root)
        u = units.loc[f'L{layer:02d}']
        cumulative_s += float(u['end'] - u['start'])
        loads.append(float(u['loop_start'] - u['start']))
        evals_s.append(float(u['end'] - u['loop_start']) / per_layer)
        rows.append({'pipeline': 'newton', 'layer': layer, 'ppl': ppl, 'step': previous - ppl,
                     'cumulative_evals': layer * per_layer, 'cumulative_compute_s': cumulative_s,
                     'side': side, 'anchor': anchor, 'accepted': previous - ppl > 0,
                     'stop_reason': None, 'searched': True})
        previous = ppl
    curve = pd.DataFrame(rows)
    run = {
        'pipeline': 'newton', 'label': LABELS['newton'], 'gpus': GPUS,
        'coverage': 'softmax', 'baseline': None, 'start_ppl': uniform, 'final_ppl': previous,
        'evals': N_LAYERS * per_layer, 'loads_as_designed': N_LAYERS, 'loads_as_run': N_LAYERS,
        'load_s_median': statistics.median(loads), 'eval_s_median': statistics.median(evals_s),
        'compute_s': cumulative_s, 'load_s_total': sum(loads),
        'eval_s_total': cumulative_s - sum(loads),
        'manual_steps': N_LAYERS - 1, 'candidates_per_layer': per_layer,
    }
    return run, curve


def load_artifact(lab, times, allow_partial=False):
    root = os.path.join(lab, 'artifact_csv')
    frames, missing = [], []
    for split, chunks in EXPECT.items():
        merged = os.path.join(root, f'nonlinear_config_split_{split}', 'metric.csv')
        if os.path.exists(merged):
            df = pd.read_csv(merged)
            check(len(df) == sum(chunks.values()), f'split_{split} has {len(df)} rows')
            frames.append(df.assign(split=split))
            continue
        for chunk, n in chunks.items():
            path = os.path.join(root, f'nonlinear_config_split_{split}_{chunk}', 'metric.csv')
            if not os.path.exists(path):
                missing.append(f'split_{split}_{chunk}')
                continue
            df = pd.read_csv(path)
            check(len(df) == n, f'split_{split}_{chunk} has {len(df)} rows, expected {n}')
            frames.append(df.assign(split=split))
    e2e_path = os.path.join(root, 'nonlinear_e2e_config', 'metric.csv')
    if not os.path.exists(e2e_path):
        missing.append('e2e')
    check(allow_partial or not missing,
          f'Paper artifact incomplete, missing {missing}; rerun when it finishes or pass '
          f'--allow_partial_artifact to plot what exists')
    sweep = pd.concat(frames, ignore_index=True)
    if 'e2e' in missing:
        e2e = pd.DataFrame(columns=['function_name', 'value'])
    else:
        e2e = pd.read_csv(e2e_path)
        check(len(e2e) == 3, f'end to end has {len(e2e)} rows')

    baseline = float(sweep.loc[sweep['function_name'] == 'torch', 'value'].iloc[0])
    patched = sweep[sweep['function_name'] != 'torch']
    flat = {'baseline': baseline, **{f'row {i}': float(v) for i, v in enumerate(patched['value'])},
            **{f"end to end {r['function_name']}": float(r['value']) for _, r in e2e.iterrows()}}
    warnings = unpatched_warnings(flat)
    check(not warnings, f'{len(warnings)} artifact results equal the unpatched baseline')

    units = latest_success(times, 'ae')
    check(len(units) == 13 - len(missing),
          f'artifact has {len(units)} timed units, expected {13 - len(missing)}')
    loads = (units['loop_start'] - units['start']).astype(float)
    per_eval = [(r['end'] - r['loop_start']) / (3 if r['unit'] == 'e2e' else
                                                EXPECT[r['unit'].split('_')[1]][r['unit'].split('_')[2]])
                for _, r in units.iterrows()]

    vlp_sm = patched[(patched['function_name'] == 'vlp') & patched['patch_attention']
                     & ~patched['patch_ffn']]
    best_sm = vlp_sm.loc[vlp_sm['value'].idxmin()]
    e2e_by = e2e.set_index('function_name')['value'].astype(float)

    run = {
        'pipeline': 'artifact', 'gpus': GPUS,
        'label': LABELS['artifact'] + (f", partial: {', '.join(missing)} not yet run"
                                                     if missing else ''),
        'coverage': 'softmax, SiLU, both', 'baseline': baseline, 'start_ppl': None,
        'final_ppl': None, 'evals': len(sweep) + len(e2e), 'loads_as_designed': 3,
        'loads_as_run': len(units), 'load_s_median': float(loads.median()),
        'eval_s_median': float(statistics.median(per_eval)),
        'compute_s': float((units['end'] - units['start']).sum()),
        'load_s_total': float(loads.sum()),
        'eval_s_total': float((units['end'] - units['loop_start']).sum()), 'manual_steps': 0,
        'candidates_per_layer': None,
    }
    refs = {
        'best_uniform_vlp_softmax': float(best_sm['value']),
        'best_uniform_vlp_softmax_params': {k: best_sm[k] for k in
                                            ('attn_exp_dim', 'attn_max_exp', 'attn_min_exp',
                                             'attn_lut_build')},
        'e2e': {k: float(v) for k, v in e2e_by.items()},
        'missing': missing,
    }
    return run, sweep, e2e, refs


GRID_WINDOWS = (('max', 1), ('max', 2), ('max', 3))


def newton_grid_ablation(lab):
    root = os.path.join(lab, 'newton_csv')
    rows, previous = [], None
    for layer in range(1, N_LAYERS + 1):
        df = newton_next.candidates(layer, root)
        anchor, side, ppl = newton_next.pick(layer, root)
        in_grid = df[[(s, a) in GRID_WINDOWS for s, a in zip(df['lut_build'], df['anchor'])]]
        grid_best = in_grid.loc[in_grid['value'].idxmin()]
        if previous is None:
            previous = float(df[(df['lut_build'] == 'max') & (df['anchor'] == 2)]['value'].iloc[0])
        rows.append({'layer': layer, 'pick': f'{side}{anchor:+d}',
                     'grid_pick': f"{grid_best['lut_build']}{int(grid_best['anchor']):+d}",
                     'start_ppl': previous, 'full_ppl': ppl, 'grid_ppl': float(grid_best['value']),
                     'full_gain': previous - ppl, 'grid_gain': previous - float(grid_best['value']),
                     'shortfall': float(grid_best['value']) - ppl})
        previous = ppl
    out = pd.DataFrame(rows)
    out['full_cumulative'] = out['full_gain'].cumsum()
    out['grid_cumulative'] = out['grid_gain'].cumsum()
    return out


def newton_landscape(lab):
    root = os.path.join(lab, 'newton_csv')
    rows, picks = [], []
    for layer in range(1, N_LAYERS + 1):
        df = newton_next.candidates(layer, root)
        rows.append((df['value'] - df['value'].min()).to_numpy(dtype=float))
        picks.append(int(df['value'].to_numpy().argmin()))
    labels = ([f'max{v:+d}' for v in newton_next.MAX_VALUES]
              + [f'min{v:+d}' for v in newton_next.MIN_VALUES])
    return np.vstack(rows), picks, labels


def budget_point(curve, column, budget):
    reached = curve[curve[column] <= budget]
    row = reached.iloc[-1]
    return float(curve['ppl'].iloc[0] - row['ppl']), int(row['layer'])


def finish_runs(runs, baseline):
    for run in runs:
        run['baseline'] = baseline
        if run['start_ppl'] is not None:
            run['delta'] = run['start_ppl'] - run['final_ppl']
            run['loss_recovered'] = run['delta'] / (run['start_ppl'] - baseline)
        else:
            run['delta'] = run['loss_recovered'] = None
        run['compute_min'] = run['compute_s'] / 60
        run['load_min'] = run['load_s_total'] / 60
        run['eval_min'] = run['eval_s_total'] / 60
        run['gpu_hours'] = run['compute_s'] * run['gpus'] / 3600
    return pd.DataFrame(runs)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def style_axis(ax):
    ax.tick_params(length=0)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASELINE)


def end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', va='center',
                ha='left', fontsize=10, color=INK2)
    ax.plot([x], [y], 'o', ms=6, color=color, mec=SURFACE, mew=2, zorder=5)


def fig_layer_curve(curves, runs, refs, baseline, path):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1.1], 'hspace': 0.08})
    by = runs.set_index('pipeline')
    names = [n for n in SEARCH_RUNS if n in set(curves['pipeline'])]
    start_dy = {'newest': 7, 'newton': -16, 'newest_fixed': 7}
    for i, name in enumerate(names):
        color, arrow_x = COLORS[name], 34.2 + 2.4 * i
        c = curves[curves['pipeline'] == name].reset_index(drop=True)
        start, final = float(c['ppl'].iloc[0]), float(c['ppl'].iloc[-1])
        drop = start - final
        searched = c[c['searched']]
        last = int(searched.index[-1])
        skipped = c.loc[last + 1:, 'layer']
        note = ''
        if len(skipped):
            first, end = int(skipped.iloc[0]), int(skipped.iloc[-1])
            note = (f', layer {first} not searched' if first == end
                    else f', layers {first}–{end} not searched')
        ax.plot(c['layer'].iloc[:last + 1], c['ppl'].iloc[:last + 1], color=color, lw=2,
                solid_capstyle='round',
                label=f"{DISPLAY[name]}: {start:.3f} → {final:.3f}, "
                      f"decreased {drop:.3f}{note}")
        if len(skipped):
            ax.plot(c['layer'].iloc[last:], c['ppl'].iloc[last:], color=color, lw=1.5,
                    linestyle=(0, (2, 2)))
        ax.plot([0], [start], 'o', ms=6, color=color, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f'start {start:.3f}', (0, start), xytext=(8, start_dy[name]),
                    textcoords='offset points', fontsize=10, color=INK2)
        end_label(ax, c['layer'].iloc[-1], final, f'{final:.3f}', color)
        ax.plot([arrow_x - 0.6, arrow_x], [start, start], color=color, lw=1.2)
        ax.annotate('', xy=(arrow_x, final), xytext=(arrow_x, start),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.8, shrinkA=0, shrinkB=0))
        ax.text(arrow_x + 0.25, (start + final) / 2, f'−{drop:.3f}', va='center',
                ha='left', fontsize=11, color=INK, weight='bold')
    ax.set_xlim(-0.8, 34.2 + 2.4 * len(names) + 1.6)

    references = [(baseline, 'Unpatched (floating-point) model', BASELINE, (0, (1, 0)), 'top')]
    if refs:
        references += [
            (refs['best_uniform_vlp_softmax'],
             'Paper artifact, best uniform window, softmax only, unfixed kernel',
             ARTIFACT, (0, (4, 3)), 'top'),
            (refs['e2e'].get('vlp'), 'Paper artifact, end to end VLP, softmax + SiLU, unfixed kernel',
             ARTIFACT, (0, (1, 2)), 'top'),
        ]
    for value, text, color, dash, va in references:
        if value is None:
            continue
        if color == ARTIFACT:
            ax.axhline(value, color=color, lw=1.5, linestyle=dash, zorder=1,
                       label=f'{text}: {value:.3f}')
            continue
        ax.axhline(value, color=color, lw=1.5, linestyle=dash, zorder=1)
        ax.text(N_LAYERS - 0.2, value, f'{text}  {value:.3f}', va=va, ha='right',
                fontsize=9.5, color=INK2)

    low, high = ax.get_ylim()
    ax.set_ylim(min(low, baseline - 0.06 * (high - low)), high)
    ax.set_ylabel(PPL_AXIS)
    ax.set_title('Perplexity after each layer is locked', pad=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.1), fontsize=10)
    style_axis(ax)

    width = 0.8 / len(names)
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        c = curves[(curves['pipeline'] == name) & (curves['layer'] > 0)]
        bx.bar(c['layer'] + offset, c['step'].clip(lower=0), width=width, color=COLORS[name],
               linewidth=0)
    bx.set_ylabel('PPL reduction\nat this layer')
    bx.set_xlabel(LAYER_AXIS)
    bx.set_xticks(range(0, N_LAYERS + 1, 4))
    style_axis(bx)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def draw_search_line(ax, c, y, color, label=None):
    c = c.reset_index(drop=True)
    y = y.reset_index(drop=True)
    last = int(c[c['searched']].index[-1])
    ax.plot(c['layer'].iloc[:last + 1], y.iloc[:last + 1], color=color, lw=2,
            solid_capstyle='round', label=label)
    if last < len(c) - 1:
        ax.plot(c['layer'].iloc[last:], y.iloc[last:], color=color, lw=1.5,
                linestyle=(0, (2, 2)))
    ax.plot([0], [y.iloc[0]], 'o', ms=6, color=color, mec=SURFACE, mew=2, zorder=5)


def ref_line(ax, value, text, color, dash, va='bottom'):
    ax.axhline(value, color=color, lw=1.4, linestyle=dash, zorder=1)
    ax.text(N_LAYERS - 0.2, value, f'{text}  {value:.3f}', va=va, ha='right', fontsize=9.5,
            color=INK2)


def fig_layer_curve_shared(curves, baseline, path):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for name in [n for n in ('newest', 'newton', 'newest_fixed') if n in set(curves['pipeline'])]:
        c = curves[curves['pipeline'] == name]
        start, final = float(c['ppl'].iloc[0]), float(c['ppl'].iloc[-1])
        draw_search_line(ax, c, c['ppl'], COLORS[name], f'{DISPLAY[name]}  {start:.3f} → {final:.3f}')
        end_label(ax, c['layer'].iloc[-1], final, f'{final:.3f}', COLORS[name])
    ref_line(ax, baseline, 'Unpatched', BASELINE, (0, (1, 0)), 'top')
    low, high = ax.get_ylim()
    ax.set_ylim(baseline - 0.08 * (high - baseline), high)
    ax.set_ylabel(PPL_AXIS)
    ax.legend(loc='lower right', bbox_to_anchor=(0.97, 0.16), fontsize=10)
    ax.set_xlabel(LAYER_AXIS)
    ax.set_xticks(range(0, N_LAYERS + 1, 4))
    ax.set_xlim(-0.8, N_LAYERS + 3)
    style_axis(ax)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plural(n, word):
    return f"{n} {word}{'s' if n != 1 else ''}"


def fig_cost_breakdown(runs, refs, path):
    by = runs.set_index('pipeline')
    order = [p for p in ('newest', 'newest_fixed', 'newton', 'artifact') if p in by.index]
    rows = list(enumerate(order))
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.85 * len(order)))
    for i, name in rows:
        r = by.loc[name]
        ax.barh(i, r['load_min'], color=BASELINE, height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.barh(i, r['eval_min'], left=r['load_min'], height=0.6, edgecolor=SURFACE, linewidth=2,
                color=COLORS[name])
        if name == 'artifact':
            ppl = f"PPL {refs['best_uniform_vlp_softmax']:.3f}" if refs else ''
        else:
            ppl = f"PPL {r['final_ppl']:.3f}"
        ax.text(r['load_min'] + r['eval_min'] / 2, i, ppl, ha='center', va='center', fontsize=10,
                color='#ffffff', weight='bold')
        ax.text(r['compute_min'] + 1.5, i,
                f"{r['compute_min']:.0f} min, {plural(int(r['gpus']), 'GPU')}, "
                f"{plural(int(r['loads_as_run']), 'load')}",
                va='center', fontsize=10, color=INK2)
    ax.set_yticks(range(len(order)), [DISPLAY[n] for n in order])
    ax.invert_yaxis()
    ax.set_xlabel('Minutes')
    ax.set_xlim(0, by['compute_min'].max() * 1.25)
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)
    ax.set_title('Compute per search', pad=12)
    handles = [Rectangle((0, 0), 1, 1, fc=BASELINE),
               tuple(Rectangle((0, 0), 1, 1, fc=COLORS[n]) for _, n in rows)]
    ax.legend(handles, ['Model load', 'Evaluation'],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='lower right', fontsize=10)
    style_axis(ax)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_grid_ablation(ablation, path):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1.1], 'hspace': 0.08})
    x = [0] + list(ablation['layer'])
    full = [0.0] + list(ablation['full_cumulative'])
    grid = [0.0] + list(ablation['grid_cumulative'])
    ax.plot(x, full, color=NEWTON, lw=2, label='All 16 windows (newton picks)')
    ax.plot(x, grid, color=NEWTON, lw=2, linestyle=(0, (4, 3)), label='max+1..max+3 only (Newest grid)')
    end_label(ax, x[-1], full[-1], f'{full[-1]:.3f}', NEWTON, dy=7)
    end_label(ax, x[-1], grid[-1], f'{grid[-1]:.3f}  ({grid[-1] / full[-1]:.0%})', NEWTON, dy=-7)
    ax.set_ylabel('Cumulative PPL reduction')
    ax.set_title('3-window grid vs all 16 (newton data, unfixed kernel)', pad=12)
    ax.legend(loc='lower right', fontsize=10)
    style_axis(ax)
    bx.bar(ablation['layer'], ablation['shortfall'], width=0.6, color=NEWTON, linewidth=0)
    bx.set_ylabel('Shortfall (PPL)')
    bx.set_xlabel(LAYER_AXIS)
    bx.set_xticks(range(0, N_LAYERS + 1, 4))
    style_axis(bx)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_landscape(matrix, picks, labels, path):
    cmap = LinearSegmentedColormap.from_list('seq', RAMP)
    fig, ax = plt.subplots(figsize=(10, 11))
    shown = np.maximum(matrix, 1e-6)
    image = ax.imshow(shown, cmap=cmap, norm=LogNorm(vmin=1e-6, vmax=max(1e-3, shown.max())),
                      aspect='auto', interpolation='nearest')
    for layer, col in enumerate(picks):
        ax.add_patch(Rectangle((col - 0.5, layer - 0.5), 1, 1, fill=False, ec=HIGHLIGHT, lw=2))
    grid_cols = [labels.index(f'{s}{a:+d}') for s, a in GRID_WINDOWS]
    ax.add_patch(Rectangle((min(grid_cols) - 0.5, -0.5), len(grid_cols), N_LAYERS, fill=False,
                           ec=INK2, lw=1.2, linestyle=(0, (3, 2))))
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(N_LAYERS), [str(i) for i in range(1, N_LAYERS + 1)], fontsize=8.5)
    ax.set_ylabel('Layer')
    ax.set_xlabel('Candidate window (anchor side and exponent)')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label('PPL above layer best (log)')
    bar.outline.set_visible(False)
    ax.set_title('newton candidates per layer (unfixed kernel)', pad=12)
    ax.text(0, -0.1, 'Orange = pick.  Dashed = Newest grid.', transform=ax.transAxes, fontsize=9,
            color=MUTED)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def heat_panel(ax, sweep, title, filters, row, col, ylabel, xlabel, cmap):
    df = sweep
    for key, value in filters.items():
        df = df[df[key] == value]
    table = df.pivot_table(index=row, columns=col, values='value', aggfunc='last', dropna=False)
    table = table.sort_index().sort_index(axis=1)
    values = table.to_numpy(dtype=float)
    best = np.nanmin(values)
    masked = np.where(values > 2 * best, np.nan, values)
    lo = best
    hi = float(np.nanpercentile(masked, 90))
    hi = hi if hi > lo else lo + 1e-9
    image = ax.imshow(masked, cmap=cmap, aspect='auto', interpolation='nearest', vmin=lo, vmax=hi)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label('PPL', fontsize=8, color=INK2)
    bar.ax.tick_params(labelsize=7, length=0)
    bar.outline.set_visible(False)
    size = 7.5 if values.shape[1] <= 7 else 5.6
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(masked[i, j]):
                continue
            dark = (v - lo) / (hi - lo) > 0.55
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=size,
                    color='#ffffff' if dark else INK)
    bi, bj = np.unravel_index(np.nanargmin(values), values.shape)
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=HIGHLIGHT, lw=2.5))
    ax.set_xticks(range(values.shape[1]), [f'{int(c)}' for c in table.columns], fontsize=8)
    ax.set_yticks(range(values.shape[0]), [f'{int(r)}' for r in table.index], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{title}  (best {best:.3f})', fontsize=10.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return best


def fig_heatmaps(sweep, e2e, baseline, path, missing=()):
    cmap = LinearSegmentedColormap.from_list('seq', RAMP)
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), gridspec_kw={'hspace': 0.55, 'wspace': 0.5})
    flat = axes.ravel()
    for ax, spec in zip(flat, HEATMAPS):
        heat_panel(ax, sweep, *spec, cmap=cmap)

    tx = flat[-1]
    tx.axis('off')
    by = e2e.set_index('function_name')['value'].astype(float)
    rows = [('Base', baseline), ('VLP', by.get('vlp')), ('PWL', by.get('pwl')),
            ('T', by.get('taylor'))]
    cell = [[name, 'pending' if v is None or pd.isna(v) else f'{v:.3f}',
             f'{PAPER_E2E_100DOC[name]:.2f}'] for name, v in rows]
    table = tx.table(cellText=cell, colLabels=['', '8 docs', 'Paper (100 docs)'],
                     colWidths=[0.2, 0.32, 0.48], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (r, c), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor(GRID)
        cell_obj.set_facecolor(SURFACE)
        if r == 0:
            cell_obj.set_text_props(color=INK2, weight='bold')
    tx.set_title('End to end (T = softmax only)', fontsize=10.5)
    fig.suptitle('Paper artifact uniform sweeps, 8 docs (PPL, lower = better)\n'
                 'VLP softmax and VLP end to end on the unfixed kernel (mask leak).  '
                 'Outlined = best.  Blank = above 2× best or NaN.'
                 + (f"\nPartial ({', '.join(missing)} not yet run)" if missing else ''),
                 x=0.01, y=1.03, ha='left', fontsize=12, color=INK2)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_windows(curves, path):
    names = [n for n in SEARCH_RUNS if n in set(curves['pipeline'])]
    fig, ax = plt.subplots(figsize=(14, 1.3 + 0.65 * len(names)))
    any_skipped = False
    for row, name in enumerate(names):
        c = curves[(curves['pipeline'] == name) & (curves['layer'] > 0)]
        for _, r in c.iterrows():
            x = int(r['layer'])
            if not r['searched']:
                any_skipped = True
                ax.add_patch(Rectangle((x - 0.46, row - 0.42), 0.92, 0.84, fc=SURFACE,
                                       ec=BASELINE, lw=1, linestyle=(0, (2, 2))))
                continue
            ax.add_patch(Rectangle((x - 0.46, row - 0.42), 0.92, 0.84,
                                   fc=SIDE_FILL[r['side']], ec=SURFACE, lw=2))
            ax.text(x, row, f"{int(r['anchor']):+d}", ha='center', va='center', fontsize=8.5,
                    color=INK)
    ax.set_xlim(0.4, N_LAYERS + 0.6)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(range(len(names)), [DISPLAY[n] for n in names])
    ax.set_xticks(range(1, N_LAYERS + 1), [str(i) if i % 4 == 0 or i == 1 else ''
                                           for i in range(1, N_LAYERS + 1)])
    ax.set_xlabel('Layer')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    handles = [Rectangle((0, 0), 1, 1, fc=SIDE_FILL[s], ec=BASELINE) for s in ('max', 'min')]
    texts = ['max-anchored', 'min-anchored']
    if any_skipped:
        handles.append(Rectangle((0, 0), 1, 1, fc=SURFACE, ec=BASELINE, linestyle=(0, (2, 2))))
        texts.append('not searched')
    ax.legend(handles, texts, loc='upper left', bbox_to_anchor=(0, -0.3), ncol=len(texts),
              fontsize=9.5)
    ax.text(0, -0.5 if len(names) > 2 else -0.62,
            'Number = anchor exponent.  LUT size 10 exponents (Newest), 16 (newton).',
            transform=ax.transAxes, fontsize=9, color=MUTED, va='top')
    ax.set_title('Window per layer', pad=10)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--lab', required=True)
    p.add_argument('--newest', required=True)
    p.add_argument('--newest_tag', default='paired-full32-v2')
    p.add_argument('--newest_fixed_tag', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--allow_partial_artifact', action='store_true')
    args = p.parse_args(argv)
    if args.out is None:
        args.out = 'output/original_compare_maskfix_v2' if args.newest_fixed_tag else 'output/original_compare'

    times = pd.read_csv(os.path.join(args.lab, 'orig_compare', 'unit_times.csv'))
    newest, newest_curve, newest_trace = load_newest(args.newest, args.newest_tag, expect_evals=65)
    fixed = fixed_curve = None
    if args.newest_fixed_tag:
        fixed, fixed_curve, _ = load_newest(args.newest, args.newest_fixed_tag, 'newest_fixed',
                                            expect_evals=65)
        check(abs(fixed['baseline'] - newest['baseline']) < 1e-6,
              f"baselines differ: fixed kernel run {fixed['baseline']} vs newest edition {newest['baseline']}")
    newton, newton_curve = load_newton(args.lab, times)

    artifact_root = os.path.join(args.lab, 'artifact_csv')
    has_artifact = os.path.isdir(artifact_root) and any(
        d.startswith('nonlinear_config_split_') for d in os.listdir(artifact_root))
    baseline = newest['baseline']
    artifact = sweep = e2e = refs = None
    if has_artifact:
        artifact, sweep, e2e, refs = load_artifact(args.lab, times, args.allow_partial_artifact)
        if refs['missing']:
            print(f"PARTIAL Paper artifact: {', '.join(refs['missing'])} not yet run")
        check(abs(artifact['baseline'] - baseline) < 1e-6,
              f"baselines differ: paper artifact {artifact['baseline']} vs newest edition {baseline}")
        baseline = artifact['baseline']
    else:
        print('no artifact_csv/ yet: Paper artifact references, heatmaps and cost row skipped')

    runs = finish_runs([r for r in (newest, fixed, newton, artifact) if r is not None], baseline)
    curves = pd.concat([c for c in (newest_curve, fixed_curve, newton_curve) if c is not None],
                       ignore_index=True)

    ablation = newton_grid_ablation(args.lab)
    check(abs(ablation['full_cumulative'].iloc[-1] - (newton['start_ppl'] - newton['final_ppl'])) < 1e-9,
          'grid ablation full gain does not match the newton edition curve')
    by_run = runs.set_index('pipeline')
    for column, key in (('cumulative_evals', 'evals'), ('cumulative_compute_s', 'compute_s')):
        gain, layer = budget_point(newton_curve, column, float(by_run.loc['newest', key]))
        runs.loc[runs['pipeline'] == 'newton', f'gain_at_newest_budget_{key}'] = gain
        runs.loc[runs['pipeline'] == 'newton', f'layer_at_newest_budget_{key}'] = layer
    check(np.allclose(runs['load_min'] + runs['eval_min'], runs['compute_min']),
          'load + eval minutes do not add up to compute minutes')

    figures = os.path.join(args.out, 'figures')
    os.makedirs(figures, exist_ok=True)
    runs.to_csv(os.path.join(args.out, 'runs.csv'), index=False)
    curves.to_csv(os.path.join(args.out, 'layer_curve.csv'), index=False)
    ablation.to_csv(os.path.join(args.out, 'grid_ablation.csv'), index=False)
    if has_artifact:
        parts = [sweep] + ([e2e.assign(split='e2e')] if len(e2e) else [])
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(args.out, 'ae_sweep.csv'), index=False)
        with open(os.path.join(args.out, 'references.yaml'), 'w') as f:
            yaml.safe_dump(plain(refs), f, sort_keys=False)

    if fixed is not None:
        fig_layer_curve_shared(curves, baseline, os.path.join(figures, 'per_layer'))
    else:
        fig_layer_curve(curves, runs, refs, baseline, os.path.join(figures, 'per_layer'))
    if has_artifact:
        fig_heatmaps(sweep, e2e, baseline, os.path.join(figures, 'uniform_sweep'), refs['missing'])
    fig_windows(curves, os.path.join(figures, 'windows'))
    fig_cost_breakdown(runs, refs, os.path.join(figures, 'cost_breakdown'))
    fig_grid_ablation(ablation, os.path.join(figures, 'grid_ablation'))
    fig_landscape(*newton_landscape(args.lab), os.path.join(figures, 'candidate_landscape'))

    cols = ['pipeline', 'coverage', 'start_ppl', 'final_ppl', 'delta', 'loss_recovered', 'evals',
            'loads_as_designed', 'loads_as_run', 'eval_s_median', 'compute_min', 'gpu_hours',
            'manual_steps']
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(runs[cols].round(4).to_string(index=False))
    print(f'baseline {baseline:.6f}')
    full, grid = ablation['full_cumulative'].iloc[-1], ablation['grid_cumulative'].iloc[-1]
    print(f'grid ablation: all 16 windows {full:.4f}, max+1..max+3 only {grid:.4f} '
          f'({grid / full:.1%} kept, shortfall {full - grid:.4f})')
    nr = runs.set_index('pipeline').loc['newton']
    print(f"newton edition at the Newest edition budget: "
          f"{nr['gain_at_newest_budget_evals']:.4f} after {int(nr['layer_at_newest_budget_evals'])} "
          f"layers (same evals), {nr['gain_at_newest_budget_compute_s']:.4f} after "
          f"{int(nr['layer_at_newest_budget_compute_s'])} layers (same minutes)")
    print(f'wrote {os.path.abspath(args.out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

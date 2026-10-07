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
NEWTON = '#eb6834'
ARTIFACT = '#1baf7a'
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

HEATMAPS = [
    ('VLP softmax, max-built', dict(function_name='vlp', patch_attention=True, patch_ffn=False,
                                    attn_lut_build='max'), 'attn_exp_dim', 'attn_max_exp',
     'LUT size (exp_dim)', 'max exp'),
    ('VLP softmax, min-built', dict(function_name='vlp', patch_attention=True, patch_ffn=False,
                                    attn_lut_build='min'), 'attn_exp_dim', 'attn_min_exp',
     'LUT size (exp_dim)', 'min exp'),
    ('VLP SiLU, max-built', dict(function_name='vlp', patch_attention=False, patch_ffn=True,
                                 ffn_lut_build='max'), 'ffn_exp_dim', 'ffn_max_pos_exp',
     'LUT size (exp_dim)', 'max exp'),
    ('VLP SiLU, min-built', dict(function_name='vlp', patch_attention=False, patch_ffn=True,
                                 ffn_lut_build='min'), 'ffn_exp_dim', 'ffn_min_pos_exp',
     'LUT size (exp_dim)', 'min exp'),
    ('PWL softmax', dict(function_name='pwl', patch_attention=True, patch_ffn=False),
     'attn_segments', 'attn_segment_0', 'segments', 'segment range'),
    ('PWL SiLU', dict(function_name='pwl', patch_attention=False, patch_ffn=True),
     'ffn_segments', 'ffn_segment_0', 'segments', 'segment range'),
    ('Taylor softmax', dict(function_name='taylor', patch_attention=True, patch_ffn=False),
     'attn_degrees', 'attn_degree_center', 'degrees', 'degree center'),
]

PAPER_E2E_100DOC = {'Base': 5.75, 'VLP': 6.21, 'PWL': 5.68, 'T': 5.78}


def check(condition, message):
    if not condition:
        raise SystemExit(f'check failed: {message}')


def latest_success(times, pipeline):
    rows = times[(times['pipeline'] == pipeline) & (times['rc'] == 0)]
    return rows.drop_duplicates(subset=['unit'], keep='last').sort_values('unit')


def load_newest(root, tag):
    base = os.path.join(root, 'output', 'window_search')
    with open(os.path.join(base, f'search_trace_{tag}_summary.yaml')) as f:
        data = yaml.safe_load(f)
    trace = pd.read_csv(os.path.join(base, f'search_trace_{tag}.csv'))
    cost = pd.read_csv(os.path.join(base, f'cost_and_noise_{tag}.csv'))
    summary = data['summary']
    curve = pd.DataFrame(data['locking_curve'])
    layers = pd.DataFrame(data['layers']).sort_values('layer')

    check(int(summary['n_evals']) == 65, f"newest edition n_evals {summary['n_evals']} != 65")
    check(len(layers) == N_LAYERS, f'newest edition has {len(layers)} layer rows')

    load_s = float(summary['load_s'])
    baseline_wall = float(cost.loc[cost['assignment_hash'] == 'baseline-unpatched', 'wall_s'].iloc[0])
    seed_wall = float(cost.loc[cost['assignment_hash'] != 'baseline-unpatched', 'wall_s'].iloc[0])
    lead_s = load_s + baseline_wall + seed_wall

    accepted = trace[(~trace['is_seed']) & (trace['accepted'])].groupby('layer').size()
    rows = [{'pipeline': 'newest', 'layer': 0, 'ppl': float(summary['seed_ppl']), 'step': 0.0,
             'cumulative_evals': 1, 'cumulative_compute_s': lead_s, 'side': None,
             'anchor': None, 'accepted': False, 'stop_reason': None}]
    by_layer = layers.set_index('layer')
    for _, r in curve[curve['n_locked'] > 0].iterrows():
        layer = int(r['layer'])
        info = by_layer.loc[layer]
        rows.append({'pipeline': 'newest', 'layer': layer + 1, 'ppl': float(r['ppl']),
                     'step': float(r['step']), 'cumulative_evals': 1 + int(r['cumulative_evals']),
                     'cumulative_compute_s': lead_s + float(r['cumulative_wall_s']),
                     'side': info['chosen_anchor_side'], 'anchor': int(info['chosen_anchor']),
                     'accepted': bool(accepted.get(layer, 0)),
                     'stop_reason': info['stop_reason']})
    curve_df = pd.DataFrame(rows)

    run = {
        'pipeline': 'newest', 'label': 'Newest edition (automated per-layer search)',
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
                 'anchor': None, 'accepted': False, 'stop_reason': None})
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
                     'stop_reason': None})
        previous = ppl
    curve = pd.DataFrame(rows)
    run = {
        'pipeline': 'newton', 'label': 'newton edition (manual per-layer)',
        'coverage': 'softmax', 'baseline': None, 'start_ppl': uniform, 'final_ppl': previous,
        'evals': N_LAYERS * per_layer, 'loads_as_designed': N_LAYERS, 'loads_as_run': N_LAYERS,
        'load_s_median': statistics.median(loads), 'eval_s_median': statistics.median(evals_s),
        'compute_s': cumulative_s, 'load_s_total': sum(loads),
        'eval_s_total': cumulative_s - sum(loads),
        'manual_steps': N_LAYERS - 1, 'candidates_per_layer': per_layer,
    }
    return run, curve


def load_artifact(lab, times):
    root = os.path.join(lab, 'artifact_csv')
    frames = []
    for split, chunks in EXPECT.items():
        path = os.path.join(root, f'nonlinear_config_split_{split}', 'metric.csv')
        df = pd.read_csv(path)
        check(len(df) == sum(chunks.values()), f'split_{split} has {len(df)} rows')
        frames.append(df.assign(split=split))
    sweep = pd.concat(frames, ignore_index=True)
    e2e = pd.read_csv(os.path.join(root, 'nonlinear_e2e_config', 'metric.csv'))
    check(len(e2e) == 3, f'end to end has {len(e2e)} rows')

    baseline = float(sweep.loc[sweep['function_name'] == 'torch', 'value'].iloc[0])
    patched = sweep[sweep['function_name'] != 'torch']
    flat = {'baseline': baseline, **{f'row {i}': float(v) for i, v in enumerate(patched['value'])},
            **{f"end to end {r['function_name']}": float(r['value']) for _, r in e2e.iterrows()}}
    warnings = unpatched_warnings(flat)
    check(not warnings, f'{len(warnings)} artifact results equal the unpatched baseline')

    units = latest_success(times, 'ae')
    check(len(units) == 13, f'artifact has {len(units)} timed units, expected 13')
    loads = (units['loop_start'] - units['start']).astype(float)
    per_eval = [(r['end'] - r['loop_start']) / (3 if r['unit'] == 'e2e' else
                                                EXPECT[r['unit'].split('_')[1]][r['unit'].split('_')[2]])
                for _, r in units.iterrows()]

    vlp_sm = patched[(patched['function_name'] == 'vlp') & patched['patch_attention']
                     & ~patched['patch_ffn']]
    best_sm = vlp_sm.loc[vlp_sm['value'].idxmin()]
    e2e_by = e2e.set_index('function_name')['value'].astype(float)

    run = {
        'pipeline': 'artifact', 'label': 'Paper artifact (uniform grid)',
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
        run['gpu_hours'] = run['compute_s'] * GPUS / 3600
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
    for (name, color), arrow_x in zip((('newest', NEWEST), ('newton', NEWTON)), (34.2, 36.6)):
        c = curves[curves['pipeline'] == name]
        start, final = float(c['ppl'].iloc[0]), float(c['ppl'].iloc[-1])
        drop = start - final
        ax.plot(c['layer'], c['ppl'], color=color, lw=2, solid_capstyle='round',
                label=f"{by.loc[name, 'label']}: {start:.3f} → {final:.3f}, "
                      f"decreased {drop:.3f}")
        ax.plot([0], [start], 'o', ms=6, color=color, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f'start {start:.3f}', (0, start), xytext=(8, 7), textcoords='offset points',
                    fontsize=10, color=INK2)
        end_label(ax, c['layer'].iloc[-1], final, f'{final:.3f}', color)
        ax.plot([arrow_x - 0.6, arrow_x], [start, start], color=color, lw=1.2)
        ax.annotate('', xy=(arrow_x, final), xytext=(arrow_x, start),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.8, shrinkA=0, shrinkB=0))
        ax.text(arrow_x + 0.25, (start + final) / 2, f'−{drop:.3f}', va='center',
                ha='left', fontsize=11, color=INK, weight='bold')
    ax.set_xlim(-0.8, 38.6)

    references = [(baseline, 'Unpatched baseline', BASELINE, (0, (1, 0)))]
    if refs:
        references += [
            (refs['best_uniform_vlp_softmax'], 'Paper artifact, best uniform window, softmax only',
             ARTIFACT, (0, (4, 3))),
            (refs['e2e'].get('vlp'), 'Paper artifact, end to end VLP, softmax + SiLU', ARTIFACT,
             (0, (1, 2))),
        ]
    for value, text, color, dash in references:
        if value is None:
            continue
        ax.axhline(value, color=color, lw=1.5, linestyle=dash, zorder=1)
        ax.text(N_LAYERS - 0.2, value, f'{text}  {value:.3f}', va='bottom', ha='right',
                fontsize=9.5, color=INK2)

    ax.set_ylabel('Perplexity (8 C4 documents)')
    ax.set_title('Perplexity after each layer is locked', pad=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.1), fontsize=10)
    style_axis(ax)

    width = 0.38
    for offset, (name, color) in zip((-width / 2, width / 2), (('newest', NEWEST), ('newton', NEWTON))):
        c = curves[(curves['pipeline'] == name) & (curves['layer'] > 0)]
        bx.bar(c['layer'] + offset, c['step'].clip(lower=0), width=width, color=color,
               linewidth=0)
    bx.set_ylabel('Gain at layer')
    bx.set_xlabel('Layer (0 = starting assignment)')
    bx.set_xticks(range(0, N_LAYERS + 1, 4))
    style_axis(bx)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_cost(curves, runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                             gridspec_kw={'wspace': 0.08})
    by = runs.set_index('pipeline')
    for ax, column, scale, xlabel in ((axes[0], 'cumulative_evals', 1, 'Cumulative evaluations'),
                                      (axes[1], 'cumulative_compute_s', 60,
                                       'Cumulative compute minutes, model loads included')):
        for name, color in (('newest', NEWEST), ('newton', NEWTON)):
            c = curves[curves['pipeline'] == name]
            gain = c['ppl'].iloc[0] - c['ppl']
            x = c[column] / scale
            ax.plot(x, gain, color=color, lw=2, drawstyle='steps-post',
                    label=by.loc[name, 'label'])
            end_label(ax, x.iloc[-1], gain.iloc[-1], f'{gain.iloc[-1]:.3f}', color)
        newest = by.loc['newest']
        budget = newest['evals'] if column == 'cumulative_evals' else newest['compute_s']
        newton_curve = curves[curves['pipeline'] == 'newton']
        gain, layer = budget_point(newton_curve, column, budget)
        ax.axvline(budget / scale, color=BASELINE, lw=1.2, linestyle=(0, (3, 3)), zorder=1)
        ax.plot([budget / scale], [gain], 'o', ms=7, color=NEWTON, mec=SURFACE, mew=2, zorder=6)
        unit = 'evals' if column == 'cumulative_evals' else 'min'
        ax.annotate(f'Newest edition budget ({budget / scale:.0f} {unit}):\n'
                    f'newton edition at layer {layer}, drop {gain:.3f}',
                    (budget / scale, gain), xytext=(10, -4), textcoords='offset points',
                    fontsize=9.5, color=INK2, va='top')
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0)
        style_axis(ax)
    axes[0].set_ylabel('Perplexity drop from own start')
    axes[0].set_title('Gain against search cost', pad=12)
    axes[0].legend(loc='lower right', fontsize=10)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_cost_breakdown(runs, path):
    order = [p for p in ('newest', 'newton', 'artifact') if p in set(runs['pipeline'])]
    by = runs.set_index('pipeline')
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.9 * len(order)))
    for i, name in enumerate(order):
        r = by.loc[name]
        ax.barh(i, r['load_min'], color=BASELINE, height=0.55, edgecolor=SURFACE, linewidth=2)
        ax.barh(i, r['eval_min'], left=r['load_min'], height=0.55, edgecolor=SURFACE, linewidth=2,
                color={'newest': NEWEST, 'newton': NEWTON, 'artifact': ARTIFACT}[name])
        total = r['load_min'] + r['eval_min']
        ax.text(total + 1, i, f"{total:.1f} min  =  {r['load_min']:.1f} load "
                f"({int(r['loads_as_run'])} x)  +  {r['eval_min']:.1f} eval "
                f"({int(r['evals'])} evals x {r['eval_s_median']:.1f} s)",
                va='center', fontsize=10, color=INK2)
    ax.set_yticks(range(len(order)), [by.loc[n, 'label'] for n in order])
    ax.invert_yaxis()
    ax.set_xlabel('Compute minutes on 2 x H200 (grey = model loading, colour = evaluation)')
    ax.set_xlim(0, by['compute_min'].max() * 1.9)
    ax.grid(axis='y', visible=False)
    ax.set_title('Where the compute goes', pad=12)
    style_axis(ax)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_grid_ablation(ablation, path):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1.1], 'hspace': 0.08})
    x = [0] + list(ablation['layer'])
    full = [0.0] + list(ablation['full_cumulative'])
    grid = [0.0] + list(ablation['grid_cumulative'])
    ax.plot(x, full, color=NEWTON, lw=2, label='All 16 windows per layer (newton edition picks)')
    ax.plot(x, grid, color=NEWTON, lw=2, linestyle=(0, (4, 3)),
            label='Only max+1, max+2, max+3 (the Newest edition grid)')
    end_label(ax, x[-1], full[-1], f'{full[-1]:.3f}', NEWTON, dy=7)
    end_label(ax, x[-1], grid[-1], f'{grid[-1]:.3f}  ({grid[-1] / full[-1]:.0%} of the gain)',
              NEWTON, dy=-7)
    ax.set_ylabel('Cumulative perplexity drop')
    ax.set_title('How much the 3-window grid gives up, measured on newton edition data', pad=12)
    ax.legend(loc='lower right', fontsize=10)
    style_axis(ax)
    bx.bar(ablation['layer'], ablation['shortfall'], width=0.6, color=NEWTON, linewidth=0)
    bx.set_ylabel('Shortfall at layer')
    bx.set_xlabel('Layer')
    bx.set_xticks(range(0, N_LAYERS + 1, 4))
    bx.text(0, -0.55, "Approximate: each layer's windows were scored with the earlier layers at the "
            "newton edition's own picks.", transform=bx.transAxes, fontsize=9, color=MUTED)
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
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label('Perplexity above the best window at that layer (log scale)')
    bar.outline.set_visible(False)
    ax.set_title('Every window the newton edition tried, per layer', pad=12)
    ax.text(0, -0.09, 'Orange = window picked. Dashed box = the Newest edition grid (max+1..max+3). '
            'Gaps under about 1e-3 are not resolvable on 8 documents.', transform=ax.transAxes,
            fontsize=9, color=MUTED)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def heat_panel(ax, sweep, title, filters, row, col, ylabel, xlabel, cmap):
    df = sweep
    for key, value in filters.items():
        df = df[df[key] == value]
    table = df.pivot_table(index=row, columns=col, values='value', aggfunc='last')
    table = table.sort_index().sort_index(axis=1)
    values = table.to_numpy(dtype=float)
    best = np.nanmin(values)
    masked = np.where(values > 2 * best, np.nan, values)
    lo = best
    hi = float(np.nanpercentile(masked, 90))
    hi = hi if hi > lo else lo + 1e-9
    ax.imshow(masked, cmap=cmap, aspect='auto', interpolation='nearest', vmin=lo, vmax=hi)
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


def fig_heatmaps(sweep, e2e, baseline, path):
    cmap = LinearSegmentedColormap.from_list('seq', RAMP)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), gridspec_kw={'hspace': 0.55, 'wspace': 0.3})
    flat = axes.ravel()
    for ax, spec in zip(flat, HEATMAPS):
        heat_panel(ax, sweep, *spec, cmap=cmap)

    tx = flat[-1]
    tx.axis('off')
    by = e2e.set_index('function_name')['value'].astype(float)
    rows = [('Base', baseline), ('VLP', by.get('vlp')), ('PWL', by.get('pwl')),
            ('T', by.get('taylor'))]
    cell = [[name, f'{v:.3f}', f'{PAPER_E2E_100DOC[name]:.2f}'] for name, v in rows]
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
    tx.set_title('End to end (softmax + SiLU; T is softmax only)', fontsize=10.5)
    fig.suptitle('Paper artifact uniform-window sweeps on the 8 documents  -  lower is better, '
                 'best cell outlined, colour saturates at the 90th percentile, '
                 'cells above 2x the best are blank', x=0.01, ha='left',
                 fontsize=12, color=INK2)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_windows(curves, path):
    fig, ax = plt.subplots(figsize=(14, 2.6))
    names = [('newest', 'Newest edition'), ('newton', 'newton edition')]
    for row, (name, label) in enumerate(names):
        c = curves[(curves['pipeline'] == name) & (curves['layer'] > 0)]
        for _, r in c.iterrows():
            x = int(r['layer'])
            ax.add_patch(Rectangle((x - 0.46, row - 0.42), 0.92, 0.84,
                                   fc=SIDE_FILL[r['side']], ec=SURFACE, lw=2))
            ax.text(x, row, f"{int(r['anchor']):+d}", ha='center', va='center', fontsize=8.5,
                    color=INK)
    ax.set_xlim(0.4, N_LAYERS + 0.6)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(range(len(names)), [label for _, label in names])
    ax.set_xticks(range(1, N_LAYERS + 1), [str(i) if i % 4 == 0 or i == 1 else ''
                                           for i in range(1, N_LAYERS + 1)])
    ax.set_xlabel('Layer')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    handles = [Rectangle((0, 0), 1, 1, fc=SIDE_FILL[s], ec=BASELINE) for s in ('max', 'min')]
    ax.legend(handles, ['anchored at max exponent', 'anchored at min exponent'],
              loc='upper left', bbox_to_anchor=(0, -0.3), ncol=2, fontsize=9.5)
    ax.set_title('Window chosen at each layer (anchor exponent)', pad=10)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--lab', required=True)
    p.add_argument('--newest', required=True)
    p.add_argument('--newest_tag', default='paired-full32-v2')
    p.add_argument('--out', default='output/original_compare')
    args = p.parse_args(argv)

    times = pd.read_csv(os.path.join(args.lab, 'orig_compare', 'unit_times.csv'))
    newest, newest_curve, newest_trace = load_newest(args.newest, args.newest_tag)
    newton, newton_curve = load_newton(args.lab, times)

    has_artifact = os.path.isdir(os.path.join(args.lab, 'artifact_csv', 'nonlinear_config_split_0'))
    baseline = newest['baseline']
    artifact = sweep = e2e = refs = None
    if has_artifact:
        artifact, sweep, e2e, refs = load_artifact(args.lab, times)
        check(abs(artifact['baseline'] - baseline) < 1e-6,
              f"baselines differ: paper artifact {artifact['baseline']} vs newest edition {baseline}")
        baseline = artifact['baseline']
    else:
        print('no artifact_csv/ yet: Paper artifact references, heatmaps and cost row skipped')

    runs = finish_runs([r for r in (newest, newton, artifact) if r is not None], baseline)
    curves = pd.concat([newest_curve, newton_curve], ignore_index=True)

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
        pd.concat([sweep, e2e.assign(split='e2e')], ignore_index=True).to_csv(
            os.path.join(args.out, 'ae_sweep.csv'), index=False)
        with open(os.path.join(args.out, 'references.yaml'), 'w') as f:
            yaml.safe_dump(plain(refs), f, sort_keys=False)

    fig_layer_curve(curves, runs, refs, baseline, os.path.join(figures, 'per_layer'))
    fig_cost(curves, runs, os.path.join(figures, 'cost'))
    if has_artifact:
        fig_heatmaps(sweep, e2e, baseline, os.path.join(figures, 'uniform_sweep'))
    fig_windows(curves, os.path.join(figures, 'windows'))
    fig_cost_breakdown(runs, os.path.join(figures, 'cost_breakdown'))
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

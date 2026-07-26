import os
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))

from exponent_bins import EXP_BIAS, N_BINS

OUT = os.path.join(REPO, 'output', 'week1_figures')
os.makedirs(OUT, exist_ok=True)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
BLUE = '#2a78d6'
CRITICAL = '#d03b3b'
GOOD = '#0ca30c'
ORANGE = '#eb6834'
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']

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
    'grid.color': GRID,
    'axes.grid': True,
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
})


def style(ax):
    ax.tick_params(length=0, labelsize=9)
    ax.set_axisbelow(True)


def fig1_bin_offset():
    torch.manual_seed(0)
    values = torch.randn(2048, 2048, dtype=torch.float16)
    mant, exp = torch.frexp(values)
    exp = torch.where(values != 0, exp + EXP_BIAS, torch.zeros_like(exp)).flatten()
    hist = torch.bincount(exp, minlength=N_BINS).numpy()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    idx = np.arange(N_BINS)
    ax.bar(idx, hist, width=1.0, color=BLUE, linewidth=0)

    ax.axvspan(1, 31, color=CRITICAL, alpha=0.13, linewidth=0)
    ax.axvspan(EXP_BIAS - 15, EXP_BIAS + 15, color=GOOD, alpha=0.13, linewidth=0)

    ymax = hist.max()
    ax.annotate('what the readers sliced\ntensor[1:31] — always empty',
                xy=(16, ymax * 0.06), xytext=(40, ymax * 0.72),
                color=CRITICAL, fontsize=9.5, ha='left',
                arrowprops=dict(arrowstyle='->', color=CRITICAL, lw=1.4))
    ax.annotate('where the data actually is\nbins 113–143 (exponents −15…+14)',
                xy=(EXP_BIAS, ymax * 0.85), xytext=(158, ymax * 0.72),
                color='#0a6b0a', fontsize=9.5, ha='left',
                arrowprops=dict(arrowstyle='->', color='#0a6b0a', lw=1.4))

    nz = np.nonzero(hist)[0]
    ax.set_xlabel('histogram bin')
    ax.set_ylabel('count')
    ax.set_title(f'Profiler wrote bins {nz.min()}–{nz.max()}; the analysis read bins 1–30',
                 fontsize=12.5, color=INK, pad=12, loc='left')
    ax.set_xlim(0, 200)
    style(ax)
    fig.tight_layout()
    p = os.path.join(OUT, '01_bin_offset.png')
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig2_zero_collision():
    torch.manual_seed(0)
    x = torch.randn(200000, dtype=torch.float32)
    mant, exp = torch.frexp(x)

    old = torch.where(exp != 0, exp + EXP_BIAS, exp).flatten()
    old_hist = torch.bincount(old, minlength=N_BINS).numpy()
    new = torch.where(x != 0, exp + EXP_BIAS, torch.zeros_like(exp)).flatten()
    new_hist = torch.bincount(new, minlength=N_BINS).numpy()

    lo = EXP_BIAS - 15
    total = len(x)
    before_pct = 100.0 * old_hist[lo:lo + 30].sum() / total
    after_pct = 100.0 * new_hist[lo:lo + 30].sum() / total

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bars = ax.barh([1, 0], [before_pct, after_pct], height=0.5,
                   color=[CRITICAL, BLUE], linewidth=0)

    for bar, val in zip(bars, [before_pct, after_pct]):
        ax.text(val - 1.5, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%',
                va='center', ha='right', fontsize=13, color=SURFACE, weight='bold')

    ax.set_yticks([1, 0])
    ax.set_yticklabels(['before', 'after'], fontsize=11, color=INK2)
    ax.set_xlim(0, 104)
    ax.set_xlabel('share of the tensor visible to the window search')
    ax.xaxis.set_major_formatter(lambda v, _: f'{v:.0f}%')
    ax.grid(axis='y', visible=False)
    ax.set_title('Nearly a third of every tensor was invisible to the search',
                 fontsize=12.5, color=INK, pad=12, loc='left')
    ax.text(0, -0.34,
            'torch.frexp returns exponent 0 for exact zeros AND for every value in [0.5, 1.0).\n'
            'Keying on the exponent put both in bin 0 — outside the analysis window — and left\n'
            'exponent 0, the middle of the search range, permanently empty.\n'
            'Measured on a standard-normal tensor; the exact share depends on the distribution.',
            transform=ax.transAxes, fontsize=9.5, color=INK2, va='top')
    style(ax)
    fig.tight_layout()
    p = os.path.join(OUT, '02_zero_collision.png')
    fig.savefig(p, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return p


def run_sweep():
    import transformers
    from profiling_api.evaluate import WindowEvalHarness
    from profiling_api.windows import SoftmaxWindow
    from test_evaluate_harness import TinyHost

    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(
            'hf-internal-testing/tiny-random-LlamaForCausalLM', attn_implementation='eager')
        m.eval()
        host = TinyHost(m, tempfile.mkdtemp())
        h = WindowEvalHarness(host)
        h.setup()

        exp_dims = [1, 2, 4, 8, 16]
        anchors = [4, 2, 0, -2, -4]
        data = {}
        for d in exp_dims:
            data[d] = [h.evaluate(h.uniform(attention=SoftmaxWindow(exp_dim=d, anchor=a))).ppl
                       for a in anchors]
        return exp_dims, anchors, data, h.baseline_ppl
    finally:
        os.chdir(cwd)


def fig3_sweep(exp_dims, anchors, data, baseline):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    for i, d in enumerate(exp_dims):
        ys = [y if y == y else np.nan for y in data[d]]
        ax.plot(anchors, ys, marker='o', markersize=6, linewidth=2,
                color=RAMP[i], label=f'exp_dim {d}',
                markeredgecolor=SURFACE, markeredgewidth=1.5)

    ax.axhline(baseline, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(anchors[0], baseline, '  fp16 baseline', fontsize=9, color=MUTED,
            va='bottom', ha='left')

    nan_anchors = [a for j, a in enumerate(anchors)
                   if all(data[d][j] != data[d][j] for d in exp_dims)]
    lo, hi = ax.get_ylim()
    if nan_anchors:
        edge = max(nan_anchors)
        ax.axvspan(edge + 0.55, min(anchors) - 0.4, color=CRITICAL, alpha=0.10, linewidth=0)
        ax.text(edge - 0.2, lo + (hi - lo) * 0.5,
                'every window here\nreturns NaN —\nthe model collapses',
                fontsize=9.5, color=CRITICAL, ha='center', va='center')
    ax.set_ylim(lo, hi)

    ax.invert_xaxis()
    ax.set_xticks(anchors)
    ax.set_xlabel('window ceiling (anchor exponent)')
    ax.set_ylabel('perplexity')
    ax.set_title('Perplexity has an optimum, then falls off a cliff',
                 fontsize=12.5, color=INK, pad=12, loc='left')
    leg = ax.legend(frameon=False, fontsize=9.5, ncol=5,
                    loc='upper center', bbox_to_anchor=(0.5, -0.16))
    for t in leg.get_texts():
        t.set_color(INK2)
    style(ax)
    fig.tight_layout()
    p = os.path.join(OUT, '03_window_sweep.png')
    fig.savefig(p, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return p


def fig4_saturation(exp_dims, anchors, data):
    j = anchors.index(0)
    ys = [data[d][j] for d in exp_dims]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(range(len(exp_dims)), ys, marker='o', markersize=8, linewidth=2.2,
            color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.8)

    floor = min(ys)
    ax.axhline(floor, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)))

    for i, (d, y) in enumerate(zip(exp_dims, ys)):
        ax.annotate(f'{y:,.1f}', (i, y), textcoords='offset points',
                    xytext=(0, 11), ha='center', fontsize=9, color=INK2)

    ax.axvspan(2.6, len(exp_dims) - 0.5, color=GOOD, alpha=0.10, linewidth=0)
    ax.text(3.5, max(ys) - (max(ys) - min(ys)) * 0.12,
            'plateau — extra LUT depth\nbuys nothing here',
            fontsize=9.5, color='#0a6b0a', ha='center')

    ax.set_xticks(range(len(exp_dims)))
    ax.set_xticklabels(exp_dims)
    ax.set_xlabel('exp_dim (LUT depth)')
    ax.set_ylabel('perplexity')
    ax.set_title('The saturation point — this is the W3 question, in miniature',
                 fontsize=12.5, color=INK, pad=12, loc='left')
    style(ax)
    fig.tight_layout()
    p = os.path.join(OUT, '04_saturation.png')
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def main():
    paths = [fig1_bin_offset(), fig2_zero_collision()]
    exp_dims, anchors, data, baseline = run_sweep()
    paths.append(fig3_sweep(exp_dims, anchors, data, baseline))
    paths.append(fig4_saturation(exp_dims, anchors, data))

    with open(os.path.join(OUT, 'sweep_data.csv'), 'w') as f:
        f.write('exp_dim,anchor,min_exp,max_exp,ppl\n')
        for d in exp_dims:
            for a, y in zip(anchors, data[d]):
                f.write(f'{d},{a},{a - (d - 1)},{a},{y}\n')

    print(f'baseline ppl: {baseline}')
    for p in paths:
        print('wrote', p)


if __name__ == '__main__':
    main()

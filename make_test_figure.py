import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, 'output', 'week1_figures')
os.makedirs(OUT, exist_ok=True)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
BLUE = '#2a78d6'

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
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
})

SUITES = [
    ('window schema', 27, 'anchor arithmetic, immutability, hashing, YAML'),
    ('reset_lut equivalence', 28, 'reset == fresh construction, bit for bit'),
    ('harness', 28, 'setup order, repeat evals, cost + provenance'),
    ('exponent bins', 15, 'writer/reader round trip, legacy dumps'),
    ('config emission', 13, 'emitted YAML constructs the real classes'),
    ('apply assignment', 12, 'per-layer windows on a patched model'),
    ('adapter targeting', 8, 'matches pre-refactor instrumentation'),
]


def main():
    names = [s[0] for s in SUITES]
    counts = [s[1] for s in SUITES]
    notes = [s[2] for s in SUITES]
    y = np.arange(len(SUITES))[::-1]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.barh(y, counts, height=0.55, color=BLUE, linewidth=0)

    for yy, c, n in zip(y, counts, notes):
        ax.text(c + 0.6, yy, f'{c}', va='center', ha='left', fontsize=10.5,
                color=INK, weight='bold')
        ax.text(c + 2.6, yy, n, va='center', ha='left', fontsize=9, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5, color=INK2)
    ax.set_xlim(0, 46)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(length=0)

    ax.set_title('129 passing, 2 skipped', fontsize=15, color=INK, pad=36, loc='left')
    ax.text(0, 1.02, 'all on a laptop — no GPU, no cluster, no network',
            transform=ax.transAxes, fontsize=10, color=MUTED, va='bottom')
    ax.text(0, -0.10, 'the 2 skips are vivit — that tiny checkpoint is not in the local cache',
            transform=ax.transAxes, fontsize=9, color=MUTED, va='top')

    fig.tight_layout()
    p = os.path.join(OUT, '05_test_suite.png')
    fig.savefig(p, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    main()

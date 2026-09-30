import argparse
import os
import shutil
import sys

import pandas as pd
import yaml

MAX_VALUES = [0, 1, 2, 3, 4, 5, 6, 7]
MIN_VALUES = [-7, -6, -5, -4, -3, -2, -1, 0]
DEFAULT = [2, 'max']
N_LAYERS = 32
CONFIG = 'config/nonlinear_config/vlp/vlp_softmax_layers_llama_2_7b.yaml'
CSV_ROOT = 'output/csv/nousresearch/llama-2-7b-hf/vlp_softmax_layers_llama_2_7b'
ORDER = {**{('max', v): i for i, v in enumerate(MAX_VALUES)},
         **{('min', v): len(MAX_VALUES) + i for i, v in enumerate(MIN_VALUES)}}


def candidates(layer, csv_root=CSV_ROOT):
    path = os.path.join(csv_root, str(layer), 'metric.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    col = f'attn_layer_{layer - 1}_max_min_exp'
    if col not in df.columns or 'lut_build' not in df.columns:
        return None
    df = df[df['function_name'].notna()].copy()
    df['anchor'] = df[col].astype(float).astype(int)
    df = df.drop_duplicates(subset=['anchor', 'lut_build'], keep='last')
    df = df[[(s, a) in ORDER for s, a in zip(df['lut_build'], df['anchor'])]]
    if len(df) < len(ORDER):
        return None
    df['order'] = [ORDER[(s, a)] for s, a in zip(df['lut_build'], df['anchor'])]
    return df.sort_values('order').reset_index(drop=True)


def pick(layer, csv_root=CSV_ROOT):
    df = candidates(layer, csv_root)
    best = df.loc[df['value'].idxmin()]
    return int(best['anchor']), str(best['lut_build']), float(best['value'])


def write_config(layer, config=CONFIG, csv_root=CSV_ROOT):
    original = config + '.orig'
    if not os.path.exists(original):
        shutil.copy(config, original)
    with open(original) as f:
        cfg = yaml.safe_load(f)

    windows = {}
    for done in range(1, layer):
        anchor, side, _ = pick(done, csv_root)
        windows[done] = [anchor, side]
    windows['max_value'] = list(MAX_VALUES)
    windows['min_value'] = list(MIN_VALUES)
    windows['default'] = list(DEFAULT)

    cfg['patch_per_layer'] = True
    cfg['patched_layer'] = layer
    cfg['parameters']['lut_build'] = 'both'
    cfg['parameters']['max_min_exp'] = windows
    with open(config, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)


def next_layer(csv_root=CSV_ROOT):
    for layer in range(1, N_LAYERS + 1):
        if candidates(layer, csv_root) is None:
            return layer
    return None


def summary(csv_root=CSV_ROOT):
    rows = []
    for layer in range(1, N_LAYERS + 1):
        df = candidates(layer, csv_root)
        if df is None:
            break
        anchor, side, ppl = pick(layer, csv_root)
        rows.append({'layer': layer, 'side': side, 'anchor': anchor, 'ppl': ppl,
                     'candidates': len(df)})
    if not rows:
        print('no completed layers')
        return
    first = candidates(1, csv_root)
    uniform = first[(first['lut_build'] == DEFAULT[1]) & (first['anchor'] == DEFAULT[0])]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"uniform start (every layer {DEFAULT[1]}{DEFAULT[0]:+d})  "
          f"{float(uniform['value'].iloc[0]):.6f}")
    print(f"after layer {rows[-1]['layer']}  {rows[-1]['ppl']:.6f}")
    print(f"evaluations  {int(out['candidates'].sum())}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['next', 'summary'])
    args = p.parse_args(argv)
    if args.command == 'summary':
        summary()
        return 0
    layer = next_layer()
    if layer is None:
        print('done')
        return 0
    write_config(layer)
    print(layer)
    return 0


if __name__ == '__main__':
    sys.exit(main())

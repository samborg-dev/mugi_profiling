import glob
import os
import sys

import pandas as pd

MODEL = 'NousResearch/Llama-2-7b-hf'
EXPECT = {
    '0': {'c1': 35, 'c2': 35, 'c3': 27, 'c4': 27, 'c5': 27, 'c6': 28},
    '1': {'c1': 1, 'c2': 25, 'c3': 22, 'c4': 33, 'c5': 25, 'c6': 25},
}


def merge(split, expect, root='csv'):
    frames, problems = [], []
    for chunk, n in expect.items():
        path = os.path.join(root, MODEL, f'nonlinear_config_split_{split}_{chunk}', 'metric.csv')
        if not os.path.exists(path):
            problems.append(f'split_{split}_{chunk}: missing {path}')
            continue
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset=[c for c in df.columns if c != 'value'], keep='last')
        if len(df) != n:
            problems.append(f'split_{split}_{chunk}: {len(df)} rows, expected {n}')
        frames.append(df)
    if problems:
        return None, problems
    merged = pd.concat(frames, ignore_index=True)
    out = os.path.join(root, MODEL, f'nonlinear_config_split_{split}', 'metric.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    merged.to_csv(out, index=False)
    return out, []


def main():
    failed = False
    for split, expect in EXPECT.items():
        out, problems = merge(split, expect)
        for p in problems:
            print(p, file=sys.stderr)
        if out is None:
            failed = True
        else:
            print(f'merged split_{split}: {sum(expect.values())} rows -> {out}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

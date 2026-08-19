"""Step 04: combine the bootstrap batches into per-coefficient intervals.

Each batch file holds, per block group, a string-encoded list of replicate
coefficients per parameter. This concatenates the batches so each block group
has its full set of replicates, then for each parameter derives:

    len_        replicate count, a check that no batch was missed
    std_        bootstrap standard error
    lo_, hi_    0.5th and 99.5th percentiles
    sig_pct_    99 percent percentile interval excludes zero

Parameters are streamed one at a time so peak memory is a single
(n_rows, n_replicates) array rather than the whole set at once.

    python 04_compile_bootstrap.py --batches bootstrap_replicates_*.csv
"""
import argparse
import ast
import glob

import numpy as np
import pandas as pd

from config import PARAM_NAMES, GEOID_COLUMN
from common import dedupe_batch_paths

LOWER_PERCENTILE = 0.5
UPPER_PERCENTILE = 99.5


def parse_list(value):
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def read_replicates(path, name, n_rows=None):
    col = f'boot_bb_{name}'
    df = pd.read_csv(path, usecols=lambda c: c in (GEOID_COLUMN, col,
                                                   f'boot_b_{name}'))
    if col not in df.columns:
        return None, df
    arr = np.asarray([parse_list(v) for v in df[col].values], dtype=float)
    if n_rows is not None and len(arr) != n_rows:
        raise ValueError(f'{path} has {len(arr)} rows, expected {n_rows}')
    return arr, df


def summarise_parameter(paths, name):
    blocks, point, geoid = [], None, None

    for path in sorted(paths):
        arr, df = read_replicates(path, name,
                                  n_rows=None if point is None else len(point))
        if arr is None:
            continue
        blocks.append(arr)
        if point is None:
            point_col = f'boot_b_{name}'
            point = (df[point_col].to_numpy(dtype=float)
                     if point_col in df.columns else None)
            if GEOID_COLUMN in df.columns:
                geoid = df[GEOID_COLUMN].astype(str).str.zfill(12).values

    if not blocks:
        return None
    replicates = np.hstack(blocks)

    lo = np.percentile(replicates, LOWER_PERCENTILE, axis=1)
    hi = np.percentile(replicates, UPPER_PERCENTILE, axis=1)

    out = {
        f'len_{name}': np.full(replicates.shape[0], replicates.shape[1]),
        f'std_{name}': replicates.std(axis=1),
        f'lo_{name}': lo,
        f'hi_{name}': hi,
        f'sig_pct_{name}': (lo > 0) | (hi < 0),
    }
    if point is not None:
        out[f'boot_b_{name}'] = point
    return out, geoid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches', nargs='+',
                   default=['bootstrap_replicates_*.csv'],
                   help='Batch files, or glob patterns.')
    p.add_argument('--out', default='bootstrap_summary.csv')
    return p.parse_args()


def main():
    args = parse_args()

    paths = []
    for pattern in args.batches:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    paths = dedupe_batch_paths(set(paths))
    print(f'{len(paths)} batch file(s)')

    columns, geoid = {}, None
    for name in PARAM_NAMES:
        print(f'  {name}', flush=True)
        result = summarise_parameter(paths, name)
        if result is None:
            continue
        stats, ids = result
        columns.update(stats)
        if geoid is None:
            geoid = ids

    if not columns:
        raise SystemExit('No parameters could be summarised.')

    combined = pd.DataFrame(columns)
    if geoid is not None:
        combined.insert(0, GEOID_COLUMN, geoid)

    counts = combined[f'len_{PARAM_NAMES[0]}']
    print(f'\nReplicates per block group: min {counts.min()}, '
          f'max {counts.max()}')
    if counts.nunique() != 1:
        print('  WARNING: replicate counts are uneven across block groups.')

    sig = combined[f'sig_pct_{PARAM_NAMES[0]}']
    print(f'Intercept significant at 99%: {sig.sum():,} of {len(combined):,} '
          f'({100 * sig.mean():.1f}%)')

    combined.to_csv(args.out, index=False)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()

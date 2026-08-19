"""Step 05: bootstrap confidence interval for the composition contribution.

Significance of the composition term is a property of the whole 14-covariate
sum, not of any single coefficient, so the sum has to be rebuilt for every
bootstrap replicate:

    composition_ik = sigma_y * sum_j ( beta_ijk * (x_ij - xbar_j) / sigma_xj )

for block group i and replicate k. Percentiles are then taken across k.

    python 05_composition_ci.py --batches bootstrap_replicates_*.csv
"""
import argparse
import ast
import glob

import numpy as np
import pandas as pd

from config import COVARIATES, PARAM_NAMES, DEPENDENT_VARIABLE, GEOID_COLUMN
from common import load_analysis_data, dedupe_batch_paths

LOWER_PERCENTILE = 0.5
UPPER_PERCENTILE = 99.5

# Replicate column to covariate, skipping the intercept.
COVARIATE_BY_REPLICATE_COLUMN = {
    f'boot_bb_{param}': covariate
    for param, covariate in zip(PARAM_NAMES[1:], COVARIATES)
}


def parse_list(value):
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def standardise_covariates(df):
    out = {}
    for replicate_col, covariate in COVARIATE_BY_REPLICATE_COLUMN.items():
        x = pd.to_numeric(df[covariate], errors='coerce').values
        sigma_x = np.nanstd(x)
        out[replicate_col] = ((x - np.nanmean(x)) / sigma_x if sigma_x > 0
                              else np.zeros(len(df)))
    return out


def composition_per_replicate(paths, standardised, sigma_y, n_rows):
    """Rebuild the composition sum for every replicate, batch by batch."""
    replicate_cols = list(COVARIATE_BY_REPLICATE_COLUMN)
    blocks = []

    for i, path in enumerate(sorted(paths), 1):
        print(f'  [{i}/{len(paths)}] {path}', flush=True)
        batch = pd.read_csv(path, usecols=replicate_cols)
        if len(batch) != n_rows:
            raise ValueError(f'{path} has {len(batch)} rows, expected {n_rows}')

        comp, n_reps = None, None
        for col in replicate_cols:
            betas = np.asarray([parse_list(v) for v in batch[col].values],
                               dtype=float)
            if n_reps is None:
                n_reps = betas.shape[1]
                comp = np.zeros((n_rows, n_reps))
                print(f'      {n_reps} replicates', flush=True)
            elif betas.shape[1] != n_reps:
                raise ValueError(f'{path}: {col} has {betas.shape[1]} '
                                 f'replicates, expected {n_reps}')
            comp += betas * standardised[col][:, None]

        blocks.append(comp * sigma_y)
        del batch, comp

    if not blocks:
        raise SystemExit('No replicates were read.')
    return np.hstack(blocks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='Final_submitted_data.csv')
    p.add_argument('--batches', nargs='+',
                   default=['bootstrap_replicates_*.csv'])
    p.add_argument('--out', default='composition_ci.csv')
    args = p.parse_args()

    paths = []
    for pattern in args.batches:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    paths = dedupe_batch_paths(set(paths))

    df = load_analysis_data(args.input)
    sigma_y = float(pd.to_numeric(df[DEPENDENT_VARIABLE]).std())
    print(f'{len(df):,} block groups, sigma_y {sigma_y:.4f}')
    print(f'{len(paths)} batch file(s)')

    standardised = standardise_covariates(df)
    comp = composition_per_replicate(paths, standardised, sigma_y, len(df))
    print(f'\ncomposition array {comp.shape}')

    ci_low = np.nanpercentile(comp, LOWER_PERCENTILE, axis=1)
    ci_high = np.nanpercentile(comp, UPPER_PERCENTILE, axis=1)
    significant = ~((ci_low <= 0) & (ci_high >= 0))

    print(f'Composition significant at 99%: {significant.sum():,} of '
          f'{len(df):,} ({100 * significant.mean():.1f}%)')

    pd.DataFrame({
        GEOID_COLUMN: df[GEOID_COLUMN].values,
        'comp_ci_low': ci_low,
        'comp_ci_high': ci_high,
        'comp_significant': significant,
    }).to_csv(args.out, index=False)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()

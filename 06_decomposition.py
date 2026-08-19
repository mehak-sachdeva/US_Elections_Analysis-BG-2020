"""Step 06: decompose the fitted vote share into place and composition parts.

    y_i = ybar + alpha_i*sigma_y
              + sum_j [ beta_ij * (x_ij - xbar_j) * sigma_y / sigma_xj ]
              + e_i

alpha_i is the local intercept and beta_ij the local coefficients, both fitted
on standardised data. Multiplying by sigma_y returns the components to
percentage points of vote share.

    python 06_decomposition.py --params country_ind_params.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from config import COVARIATES, PARAM_NAMES, DEPENDENT_VARIABLE, GEOID_COLUMN
from common import load_analysis_data

COEFFICIENT_TO_COVARIATE = dict(zip(PARAM_NAMES[1:], COVARIATES))
INTERCEPT_PARAM = PARAM_NAMES[0]


def resolve_prefix(params_df):
    """Coefficient columns are mgwr_b_ from step 02, boot_b_ from step 04."""
    for prefix in ('mgwr_b_', 'boot_b_'):
        if f'{prefix}{INTERCEPT_PARAM}' in params_df.columns:
            return prefix
    raise KeyError(f'No intercept column found in {list(params_df.columns)[:8]}')


def decompose(df, params_df):
    if len(df) != len(params_df):
        raise ValueError(f'{len(df)} data rows vs {len(params_df)} param rows')

    prefix = resolve_prefix(params_df)

    y = pd.to_numeric(df[DEPENDENT_VARIABLE], errors='coerce')
    y_bar = float(y.mean())
    sigma_y = float(y.std())

    context = (pd.to_numeric(params_df[f'{prefix}{INTERCEPT_PARAM}'],
                             errors='coerce').to_numpy() * sigma_y)

    composition = np.zeros(len(df))
    for param, covariate in COEFFICIENT_TO_COVARIATE.items():
        col = f'{prefix}{param}'
        if col not in params_df.columns:
            raise KeyError(f'Missing {col}')

        x = pd.to_numeric(df[covariate], errors='coerce')
        beta = pd.to_numeric(params_df[col], errors='coerce')
        sigma_x = float(x.std())
        if sigma_x == 0 or np.isnan(sigma_x):
            raise ValueError(f'{covariate} has zero or NaN variance')

        composition += beta.to_numpy() * ((x.to_numpy() - float(x.mean())) / sigma_x)
    composition *= sigma_y

    out = pd.DataFrame({
        GEOID_COLUMN: df[GEOID_COLUMN].values,
        'global_mean_y': y_bar,
        'context_contribution': context,
        'composition_contribution': composition,
    })
    out['predicted_y'] = (out['global_mean_y'] + out['context_contribution']
                          + out['composition_contribution'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='Final_submitted_data.csv')
    p.add_argument('--params', default='country_ind_params.csv',
                   help='Local coefficients written by step 02.')
    p.add_argument('--out', default='decomposition.csv')
    args = p.parse_args()

    if not os.path.exists(args.params):
        raise SystemExit(
            f'{args.params} not found. It is written by src/02_mgwr_fit.py, '
            'which must be run first.')

    out = decompose(load_analysis_data(args.input), pd.read_csv(args.params))

    print(f'ybar {out["global_mean_y"].iloc[0]:.2f}%')
    print(f'place-based  {out["context_contribution"].min():7.2f} to '
          f'{out["context_contribution"].max():7.2f} pp')
    print(f'composition  {out["composition_contribution"].min():7.2f} to '
          f'{out["composition_contribution"].max():7.2f} pp')

    out.to_csv(args.out, index=False)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()

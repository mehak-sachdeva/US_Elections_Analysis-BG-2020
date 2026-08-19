"""Step 03: residual bootstrap for the local coefficients.

Reconstructs fitted values from the step 02 coefficients, takes residuals
against the standardised dependent variable, resamples them with replacement,
adds them back, and refits with the bandwidths held fixed. Holding the
bandwidths fixed is what makes this feasible: each replicate skips the search
and is a refit only.

One batch per invocation, so a crash costs one batch rather than the whole run.
Ten batches of ten replicates gives the 100 used for inference:

    for i in $(seq -w 1 10); do
        python 03_bootstrap.py --n-bootstrap 10 --batch $i
    done

Writes bootstrap_replicates_<batch>.csv, holding the point estimate and the
replicate coefficients for each parameter.
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from sklearn.utils import resample

from mgwr.sel_bw import Sel_BW

from config import PARAM_NAMES, BANDWIDTHS, DEFAULT_INPUT, GEOID_COLUMN
from common import load_analysis_data, build_design, require_bandwidths


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=DEFAULT_INPUT)
    p.add_argument('--params', default='country_ind_params.csv',
                   help='Local coefficients written by step 02.')
    p.add_argument('--n-bootstrap', type=int, default=10,
                   help='Replicates in this batch.')
    p.add_argument('--batch', default='01',
                   help='Batch label. Zero pad so batches sort correctly.')
    p.add_argument('--nproc', type=int, default=mp.cpu_count())
    return p.parse_args()


def main():
    args = parse_args()
    bandwidths = require_bandwidths(BANDWIDTHS, len(PARAM_NAMES))
    pool = mp.Pool(args.nproc)

    df = load_analysis_data(args.input)
    print(f'{len(df):,} block groups', flush=True)

    if not os.path.exists(args.params):
        raise SystemExit(
            f'{args.params} not found. It is written by src/02_mgwr_fit.py, '
            'which must be run first.')
    orig = pd.read_csv(args.params, dtype={GEOID_COLUMN: str})

    # The arrays below are combined element-wise, so a row order mismatch
    # would corrupt every result without raising.
    if len(orig) != len(df):
        raise ValueError(f'{args.params} has {len(orig)} rows, '
                         f'{args.input} has {len(df)}')
    if GEOID_COLUMN in orig.columns:
        orig[GEOID_COLUMN] = orig[GEOID_COLUMN].str.zfill(12)
        if not (orig[GEOID_COLUMN].values == df[GEOID_COLUMN].values).all():
            raise ValueError(f'{args.params} and {args.input} are in '
                             'different row orders')

    original_params = np.array(
        orig[[f'mgwr_b_{n}' for n in PARAM_NAMES]].values, dtype=float)

    coords, X_std, y_std, X_cons = build_design(df)
    n_samples = X_std.shape[0]

    y_hat = (original_params * X_cons).sum(axis=1)
    resid = y_std.reshape(-1) - y_hat

    n_boot = args.n_bootstrap
    boot_coefs = np.zeros((n_boot, n_samples, X_cons.shape[1]))

    for i in range(n_boot):
        print(f'replicate {i + 1}/{n_boot}', flush=True)

        resampled = resample(resid, replace=True, n_samples=n_samples)
        y_boot = (y_hat + resampled).reshape(-1, 1)
        y_boot = (y_boot - y_boot.mean(axis=0)) / y_boot.std(axis=0)

        selector = Sel_BW(coords, y_boot, X_std, multi=True, spherical=False)
        selector.search(verbose=False, pool=pool,
                        multi_bw_min=bandwidths, multi_bw_max=bandwidths)

        boot_coefs[i, :, :] = selector.params

    out = pd.DataFrame({GEOID_COLUMN: df[GEOID_COLUMN].values})
    for j, name in enumerate(PARAM_NAMES):
        out[f'boot_b_{name}'] = original_params[:, j]
        out[f'boot_bb_{name}'] = boot_coefs[:, :, j].T.tolist()

    path = f'bootstrap_replicates_{args.batch}.csv'
    out.to_csv(path, index=False)
    print(f'Wrote {path}', flush=True)


if __name__ == '__main__':
    main()

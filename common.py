"""Data loading and design matrix construction.

Every stage builds coords, X and y through this module, so that the bandwidth
search, the fit and the bootstrap all standardise the same way. The bootstrap
in particular takes residuals against the fitted model, which is only valid if
the design matrix is identical.
"""
import hashlib
import os

import numpy as np
import pandas as pd

from config import COVARIATES, DEPENDENT_VARIABLE, COORD_COLUMNS, GEOID_COLUMN


def load_analysis_data(path):
    df = pd.read_csv(path, dtype={GEOID_COLUMN: str})
    df[GEOID_COLUMN] = df[GEOID_COLUMN].str.zfill(12)

    required = [GEOID_COLUMN, DEPENDENT_VARIABLE] + COVARIATES + list(COORD_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'{path} is missing: {missing}')

    # A null in any model column would silently corrupt the standardisation
    # of that whole covariate.
    nulls = df[required].isna().sum()
    if nulls.any():
        raise ValueError(f'Nulls in model columns:\n{nulls[nulls > 0]}')

    return df


def build_design(df):
    """Return coords, X_std, y_std, X_cons.

    X_cons is X_std with a leading column of ones, used to reconstruct fitted
    values from the local coefficients.
    """
    coords = np.array(list(zip(df[COORD_COLUMNS[0]], df[COORD_COLUMNS[1]])))

    X = np.array(df[COVARIATES].values, dtype=float)
    y = np.array(df[DEPENDENT_VARIABLE], dtype=float).reshape((-1, 1))

    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y_std = (y - y.mean(axis=0)) / y.std(axis=0)

    X_cons = np.hstack([np.ones((X_std.shape[0], 1)), X_std])
    return coords, X_std, y_std, X_cons


def require_bandwidths(bandwidths, n_params):
    if bandwidths is None:
        raise SystemExit(
            'config.BANDWIDTHS is None. Run src/01_bandwidth_search.py and '
            'paste the contents of bandwidths.txt into config.py.')
    if len(bandwidths) != n_params:
        raise SystemExit(
            f'config.BANDWIDTHS has {len(bandwidths)} entries, expected '
            f'{n_params}, one per parameter including the intercept.')
    return list(bandwidths)


def dedupe_batch_paths(paths, sample_bytes=2_000_000):
    """Drop bootstrap batch files whose contents duplicate one already listed.

    Duplicated replicates narrow a percentile interval rather than widening it,
    so counting a batch twice makes results look more certain than they are and
    leaves no trace in the output. Compares file size plus a hash of the head
    and tail, since these files run to gigabytes.
    """
    seen, kept, dropped = {}, [], []
    for path in sorted(paths):
        try:
            size = os.path.getsize(path)
            with open(path, 'rb') as fh:
                head = fh.read(sample_bytes)
                fh.seek(max(0, size - sample_bytes))
                tail = fh.read(sample_bytes)
        except OSError:
            kept.append(path)
            continue

        key = (size, hashlib.md5(head + tail).hexdigest())
        if key in seen:
            dropped.append((path, seen[key]))
        else:
            seen[key] = path
            kept.append(path)

    for path, original in dropped:
        print(f'  skipping {os.path.basename(path)}: identical to '
              f'{os.path.basename(original)}', flush=True)
    return kept

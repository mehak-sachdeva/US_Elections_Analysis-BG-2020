"""Step 02: fit MGWR at the bandwidths in config.BANDWIDTHS.

Passing the bandwidths as both multi_bw_min and multi_bw_max collapses the
golden-section search to a single evaluation, so no search is performed here.

Writes country_ind_params.csv: GEOID plus one local coefficient column per
parameter, in input row order.

    python 02_mgwr_fit.py --input Final_submitted_data.csv
"""
import argparse
import multiprocessing as mp

import pandas as pd

from mgwr.sel_bw import Sel_BW

from config import PARAM_NAMES, BANDWIDTHS, DEFAULT_INPUT, GEOID_COLUMN
from common import load_analysis_data, build_design, require_bandwidths


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=DEFAULT_INPUT)
    p.add_argument('--out', default='country_ind_params.csv')
    p.add_argument('--nproc', type=int, default=mp.cpu_count())
    return p.parse_args()


def main():
    args = parse_args()
    bandwidths = require_bandwidths(BANDWIDTHS, len(PARAM_NAMES))
    pool = mp.Pool(args.nproc)

    df = load_analysis_data(args.input)
    print(f'{len(df):,} block groups', flush=True)

    coords, X_std, y_std, _ = build_design(df)

    selector = Sel_BW(coords, y_std, X_std, multi=True, spherical=False)
    selector.search(verbose=True, pool=pool,
                    multi_bw_min=bandwidths, multi_bw_max=bandwidths)

    params = selector.params
    print(f'parameter array {params.shape}', flush=True)

    out = pd.DataFrame({GEOID_COLUMN: df[GEOID_COLUMN].values})
    for j, name in enumerate(PARAM_NAMES):
        out[f'mgwr_b_{name}'] = params[:, j]

    out.to_csv(args.out, index=False)
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()

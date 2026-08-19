"""Step 01: covariate-specific bandwidth search.

Minimises AICc under GAM backfitting to select one bandwidth per covariate,
expressed as an adaptive nearest-neighbour count. This is the most expensive
step in the pipeline by a wide margin.

Run with verbose output and watch the bandwidth vector printed at the end of
each backfitting iteration. The search stops when the score change falls below
tolerance, which can take many iterations on a national dataset.

Writes bandwidths.txt. Paste its contents into config.BANDWIDTHS before
running step 02.

    python 01_bandwidth_search.py --input Final_submitted_data.csv
"""
import argparse
import multiprocessing as mp

from mgwr.sel_bw import Sel_BW

from config import PARAM_NAMES, DEFAULT_INPUT
from common import load_analysis_data, build_design


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=DEFAULT_INPUT)
    p.add_argument('--out', default='bandwidths.txt')
    p.add_argument('--max-iter-multi', type=int, default=200,
                   help='Backfitting iteration cap.')
    p.add_argument('--nproc', type=int, default=mp.cpu_count())
    return p.parse_args()


def main():
    args = parse_args()
    pool = mp.Pool(args.nproc)

    df = load_analysis_data(args.input)
    print(f'{len(df):,} block groups', flush=True)

    coords, X_std, y_std, _ = build_design(df)

    selector = Sel_BW(coords, y_std, X_std, multi=True, spherical=False)
    selector.search(verbose=True, pool=pool, criterion='AICc',
                    max_iter_multi=args.max_iter_multi)

    bandwidths = selector.bw[0]
    print('\nBandwidths (nearest-neighbour counts):', flush=True)
    for name, bw in zip(PARAM_NAMES, bandwidths):
        print(f'  {name:15s} {bw}', flush=True)

    with open(args.out, 'w') as fh:
        fh.write(repr([float(b) for b in bandwidths]) + '\n')
    print(f'\nWrote {args.out}. Paste into config.BANDWIDTHS.', flush=True)


if __name__ == '__main__':
    main()

"""Step 08: aggregate block-group results to metropolitan areas.

Block groups are assigned to CBSAs by spatial join, then contributions are
averaged weighted by population, so a metro's score reflects the typical
resident rather than the typical block group. This produces the metro rankings
reported in Table 1.

Population and votes cast are summed rather than averaged, so a metro-level
count of votes attributable to place can be derived downstream.

Needs 2019 CBSA boundaries and block group geometry, neither of which is
included here. CBSA boundaries:
https://www2.census.gov/geo/tiger/TIGER2019/CBSA/

    python 08_metro_aggregation.py --decomposition decomposition.csv \
                                   --geometry block_groups.shp \
                                   --cbsa tl_2019_us_cbsa/tl_2019_us_cbsa.shp
"""
import argparse

import pandas as pd
import geopandas as gpd

from config import GEOID_COLUMN

EXCLUDED = [', AK', ', HI', ', PR', ', GU', ', VI', ', MP', ', AS']
DEFAULT_DEMOGRAPHICS = ['med_inc_10', 'pct_black', 'pct_hisp', 'log_popden']

# Summed, not population-weighted.
TOTALS = ['tot_pop', 'all_votes']


def filter_lower48(cbsa):
    return cbsa[~cbsa['NAMELSAD'].str.contains(
        '|'.join(EXCLUDED), case=False, na=False, regex=True)].copy()


def join_block_groups_to_metros(gdf, cbsa):
    """Join on CBSAFP, not on name.

    CBSA names carry a " Metro Area" or " Micro Area" suffix. Matching names
    across frames after stripping those gives empty joins that show up as
    all-NaN columns rather than as an error.
    """
    cbsa48 = filter_lower48(cbsa)
    if gdf.crs != cbsa48.crs:
        gdf = gdf.to_crs(cbsa48.crs)

    # Block groups have their own NAMELSAD ("Block Group 1"), which collides.
    left = gdf.drop(columns=[c for c in ('NAMELSAD',) if c in gdf.columns])

    joined = gpd.sjoin(left, cbsa48[['CBSAFP', 'NAMELSAD', 'geometry']],
                       how='inner', predicate='intersects')
    if len(joined) == 0:
        raise ValueError('Spatial join matched nothing. Check the CRSs.')
    return joined


def aggregate_to_metros(joined, value_columns, pop_column='tot_pop'):
    """Population-weighted mean of value_columns, plus summed TOTALS."""
    missing = [c for c in value_columns if c not in joined.columns]
    if missing:
        raise KeyError(f'Not present for aggregation: {missing}')

    weights = joined[pop_column].astype(float)
    frame = pd.DataFrame({'CBSAFP': joined['CBSAFP'].values,
                          'NAMELSAD': joined['NAMELSAD'].values})
    for col in value_columns:
        frame[f'_w_{col}'] = joined[col].astype(float).values * weights.values
    for col in TOTALS:
        if col in joined.columns:
            frame[col] = joined[col].astype(float).values

    agg = {f'_w_{c}': 'sum' for c in value_columns}
    agg.update({c: 'sum' for c in TOTALS if c in frame.columns})
    agg['NAMELSAD'] = 'first'

    metros = frame.groupby('CBSAFP').agg(agg).reset_index()
    metros = metros[metros[pop_column] > 0].copy()

    for col in value_columns:
        metros[col] = metros[f'_w_{col}'] / metros[pop_column]

    keep = (['CBSAFP', 'NAMELSAD']
            + [c for c in TOTALS if c in metros.columns]
            + list(value_columns))
    return metros[keep]


def build_metro_table(decomposition, analysis_data, geometry, cbsa,
                      demographics=None):
    demographics = list(demographics or DEFAULT_DEMOGRAPHICS)

    bg = (geometry[[GEOID_COLUMN, 'geometry']]
          .merge(decomposition, on=GEOID_COLUMN, how='inner')
          .merge(analysis_data.drop(columns=['geometry'], errors='ignore'),
                 on=GEOID_COLUMN, how='inner'))
    bg = gpd.GeoDataFrame(bg, geometry='geometry', crs=geometry.crs)

    joined = join_block_groups_to_metros(bg, cbsa)
    metros = aggregate_to_metros(
        joined,
        ['context_contribution', 'composition_contribution'] + demographics)

    # Constant across block groups, carried through for convenience.
    if 'global_mean_y' in decomposition.columns:
        metros['global_mean_y'] = float(decomposition['global_mean_y'].iloc[0])

    return metros.rename(columns={
        'context_contribution': 'avg_context_contribution',
        'composition_contribution': 'avg_composition_contribution',
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decomposition', default='decomposition.csv')
    p.add_argument('--input', default='Final_submitted_data.csv')
    p.add_argument('--geometry', required=True)
    p.add_argument('--cbsa', default='tl_2019_us_cbsa/tl_2019_us_cbsa.shp')
    p.add_argument('--demographics', nargs='*', default=None)
    p.add_argument('--out', default='metros.csv')
    args = p.parse_args()

    from common import load_analysis_data

    decomp = pd.read_csv(args.decomposition, dtype={GEOID_COLUMN: str})
    decomp[GEOID_COLUMN] = decomp[GEOID_COLUMN].str.zfill(12)

    geom = gpd.read_file(args.geometry)
    geom[GEOID_COLUMN] = geom[GEOID_COLUMN].astype(str).str.zfill(12)

    metros = build_metro_table(decomp, load_analysis_data(args.input), geom,
                               gpd.read_file(args.cbsa), args.demographics)
    metros.to_csv(args.out, index=False)

    print(f'{len(metros)} metros')
    print('\nMost Democratic by place:')
    print(metros.nlargest(5, 'avg_context_contribution')
          [['NAMELSAD', 'avg_context_contribution']].to_string(index=False))
    print('\nMost Republican by place:')
    print(metros.nsmallest(5, 'avg_context_contribution')
          [['NAMELSAD', 'avg_context_contribution']].to_string(index=False))
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()

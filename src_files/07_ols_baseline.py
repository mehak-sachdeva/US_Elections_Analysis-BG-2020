"""Step 07: global OLS baseline.

Fits the same specification as the MGWR, on standardised data with an
intercept, as the non-spatial benchmark reported in Table S1.

    python 07_ols_baseline.py --input Final_submitted_data.csv
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import COVARIATES, DEPENDENT_VARIABLE, DEFAULT_INPUT
from common import load_analysis_data

DISPLAY_NAMES = {
    'sex_ratio': 'Sex Ratio',
    'pct_fb': '% Foreign Born',
    'pct_18_29': '% Age 18-29',
    'pct_65_plu': '% Age 65+',
    'pct_black': '% Black',
    'pct_hisp': '% Hispanic',
    'pct_bach_or_higher': "% Bachelor's or Higher",
    'med_inc_10': 'Median Income ($1,000s)',
    'pct_manuf': '% Manufacturing',
    'log_popden': 'Log Population Density',
    'pct_third_': '% Third-Party Votes',
    'turnout_vap': 'Voter Turnout',
    'pct_medinc': '% Health Insurance',
    'GINI_index': 'GINI Index',
}


def significance_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=DEFAULT_INPUT)
    p.add_argument('--out', default='table_s1.csv')
    args = p.parse_args()

    df = load_analysis_data(args.input)

    X = df[COVARIATES].to_numpy(dtype=float)
    y = df[DEPENDENT_VARIABLE].to_numpy(dtype=float)

    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y_std = (y - y.mean(axis=0)) / y.std(axis=0)

    model = sm.OLS(y_std, sm.add_constant(X_std)).fit()

    table = pd.DataFrame({
        'Variable': ['Intercept'] + [DISPLAY_NAMES.get(v, v) for v in COVARIATES],
        'Coefficient': model.params.round(4),
        'Std. Error': model.bse.round(4),
        't-value': model.tvalues.round(2),
        'p-value': ['<0.0001' if p < 0.0001 else f'{p:.4f}' for p in model.pvalues],
        'Sig.': [significance_stars(p) for p in model.pvalues],
    })

    print(f'N = {model.nobs:,.0f}, R2 = {model.rsquared:.4f}, '
          f'adj R2 = {model.rsquared_adj:.4f}, F = {model.fvalue:,.0f}, '
          f'AIC = {model.aic:,.0f}, residual SE = {np.sqrt(model.mse_resid):.4f}')
    print(table.to_string(index=False))

    table.to_csv(args.out, index=False)
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()

# US Elections Analysis, Block Group Level, 2020

Multiscale geographically weighted regression (MGWR) of the 2020 US
presidential vote across 214,244 census block groups in the 48 contiguous
states, decomposing the Democratic vote share in each block group into a
place-based component and a population composition component.

## Data

Not included here. The compiled dataset is on Figshare:

    https://doi.org/10.6084/m9.figshare.32118136

Sources: block group 2020 presidential returns from Bryan (2022,
https://doi.org/10.7910/DVN/NKNWBX); block-level returns for South Dakota,
Kentucky and West Virginia from VEST (https://doi.org/10.7910/DVN/K7760H),
aggregated to 2019 block groups; 2019 ACS 5-year estimates, with the Gini
coefficient from the 2013 release; 2019 TIGER/Line block group boundaries.

## Specification

Dependent variable: Democratic share of the two-party vote. Fourteen ACS
covariates, listed in `src/config.py`. Covariates and the dependent variable
are standardised to mean 0 and variance 1 before calibration.

Adaptive bi-square kernel with covariate-specific bandwidths as
nearest-neighbour counts, selected by minimising AICc under GAM backfitting.

Uncertainty is estimated by a residual bootstrap: residuals are resampled with
replacement and added back to the fitted values, then the model is refit with
the bandwidths held fixed, 100 times. Significance is a 99 percent percentile
interval per block group.

## Pipeline

```
src/01_bandwidth_search.py    bandwidths.txt        -> paste into config.py
src/02_mgwr_fit.py            country_ind_params.csv
src/03_bootstrap.py           bootstrap_replicates_NN.csv   (10 batches)
src/04_compile_bootstrap.py   bootstrap_summary.csv
src/05_composition_ci.py      composition_ci.csv
src/06_decomposition.py       decomposition.csv
src/07_ols_baseline.py        table_s1.csv
src/08_metro_aggregation.py   metros.csv
```

Each step reads the previous step's output from the working directory. No
intermediate outputs are included in this repository. The `run/` scripts wrap
steps 01 to 03 with logging and batching.

```bash
pip install -r requirements.txt
cd run
bash 01_bandwidth_search.sh
# paste bandwidths.txt into src/config.py
bash 02_mgwr_fit.sh
bash 03_bootstrap.sh
cd ..
python src/04_compile_bootstrap.py
python src/05_composition_ci.py
python src/06_decomposition.py
python src/07_ols_baseline.py
python src/08_metro_aggregation.py --geometry block_groups.shp \
                                   --cbsa tl_2019_us_cbsa/tl_2019_us_cbsa.shp
```

Step 08 additionally needs 2019 CBSA boundaries
(https://www2.census.gov/geo/tiger/TIGER2019/CBSA/) and block group geometry
(https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2019&layergroup=Block+Groups).

## Requirements

Python 3.8, `mgwr` 2.1.2, `libpysal` 4.8.0. See `requirements.txt`.

Step 01 was run on a shared-memory Linux server with approximately 1 TB of RAM
and a 220-process pool, and takes weeks at this scale. Memory is the binding
constraint: the full 214,244 by 15 parameter surface and the backfitting
working arrays are held simultaneously, and each worker holds a copy of the
working data. Steps 02 and 03 hold the bandwidths fixed and are considerably
cheaper.

## License

MIT. See `LICENSE`.

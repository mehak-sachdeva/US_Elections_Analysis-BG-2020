"""Model specification, shared by every step."""

# Order fixes the column order of X and therefore of the fitted parameter
# array. Changing it requires changing PARAM_NAMES and BANDWIDTHS to match.
COVARIATES = [
    'sex_ratio',
    'pct_fb',
    'pct_18_29',
    'pct_65_plu',
    'pct_black',
    'pct_hisp',
    'pct_bach_or_higher',
    'med_inc_10',
    'pct_manuf',
    'log_popden',
    'pct_third_',
    'turnout_vap',
    'pct_medinc',
    'GINI_index',
]

DEPENDENT_VARIABLE = 'pct_dem_twoparty'
GEOID_COLUMN = 'GEOID'
COORD_COLUMNS = ('coord_x', 'coord_y')

DEFAULT_INPUT = 'Final_submitted_data.csv'

PARAM_NAMES = [
    'intercept', 'sex_ratio', 'pctfb', 'pct1829', 'pct6585', 'pctblack',
    'pcthisp', 'pctbach', 'medinc', 'pctmanu', 'popden', '3rdparty',
    'voterturnout', 'pctinsur', 'gini',
]

# Covariate-specific bandwidths, as nearest-neighbour counts, in the order of
# PARAM_NAMES. Produced by src/01_bandwidth_search.py, which minimises AICc
# under backfitting. Bandwidths are specific to a given y and X, so they must
# come from a search over the specification above.
#
# Paste the contents of bandwidths.txt here once the search has converged.
# Steps 02 and 03 refuse to run while this is None.
BANDWIDTHS = None

# The coord_x / coord_y columns hold block group centroids already projected
# to this CRS, so no reprojection is performed anywhere in the pipeline.
ANALYSIS_CRS = 'EPSG:3086'


# Two column names in the dataset do not describe their contents:
#
#   med_inc_10   median household income in THOUSANDS of dollars
#                (ACS B19013_001E divided by 1000)
#   pct_medinc   percentage of the population WITH HEALTH INSURANCE
#                (ACS B27010_001E over total population)

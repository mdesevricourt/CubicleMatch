"""The second part of the algorithm is to get the preferences from the survey and create the preference orderings over vectors."""

import logging
from functools import partial

import numpy as np
import pandas as pd
from config import bundles_path, data_path

from cubiclematch_jax.read_data import transform_row

np.random.seed(1234)
# load survey results

# find file starting with Cubicles
survey_files = list(data_path.glob("Cubicles*.csv"))
survey_file = survey_files[0]

# load survey results
survey_results = pd.read_csv(survey_file, index_col=0)
# drop first two rows

survey_results = survey_results.iloc[2:]

bundles = np.load(bundles_path)
print(bundles.shape)


transform_row_bundles = partial(transform_row, bundles=bundles)

# apply transform_row to each row
# set numpy seed

survey_results = survey_results.apply(transform_row_bundles, axis=1)

print(survey_results.head())
# save survey results as pandas dataframe to csv
survey_results.to_pickle(data_path/ "main_data.csv")
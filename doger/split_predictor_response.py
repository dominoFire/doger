#! /usr/bin/python3

import os
import pandas as pd
import doger.utils as utils


def split_predictor_response(csv_path, response_cols):
    # File checking
    assert os.path.exists(csv_path)
    base, file, ext = utils.split_path(csv_path)
    table = pd.read_csv(csv_path)

    # Columns checking
    cols = table.columns.values.tolist()
    for c in response_cols:
        assert c in cols
    predictor_cols = [c for c in cols if not c in response_cols]

    # Slice me nice
    response_values = table[response_cols]
    predictor_values = table[predictor_cols]

    # Saving to csv
    response_values.to_csv(utils.join_path(base, '{}_response'.format(file), ext), index=False, header=False)
    predictor_values.to_csv(utils.join_path(base, '{}_predictors'.format(file), ext), index=False)

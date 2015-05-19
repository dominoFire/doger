#! /usr/bin/python3

__author__ = 'fer'

import os
import pandas as pd
import doger.utils as utils
import numpy as np


def split_train_test(csv_path, train_test_ratio):
    assert os.path.exists(csv_path)
    base, file, ext = utils.split_path(csv_path)

    assert 0 <= train_test_ratio <= 1.

    # File reader
    table = pd.read_csv(csv_path)

    # Slice me nice
    # http://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
    msk = np.random.rand(len(table)) < 0.8
    train = table[msk]
    test = table[~msk]

    # Saving
    train.to_csv(utils.join_path(base, '{}_train'.format(file), ext), index=False)
    test.to_csv(utils.join_path(base, '{}_test'.format(file), ext), index=False)

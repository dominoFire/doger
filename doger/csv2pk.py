__author__ = '@dominofire'

from doger.utils import split_path, join_path, save_object
import pandas as pd
import os


def convert_csv2pk(tdm_csv, y_csv):
    """
    Convert
    :param tdm_csv:
    :param y_csv:
    :return:
    """
    assert os.path.exists(tdm_csv) and os.path.exists(y_csv) and os.path.isfile(tdm_csv) and os.path.isfile(y_csv)

    tdm_t = split_path(tdm_csv)
    df = pd.read_csv(tdm_csv)
    y_t = split_path(y_csv)
    y = pd.read_csv(y_csv, header=None)

    assert df.shape[0] == y.shape[0] and y.shape[1] == 1

    y = y.ix[:, 0].values

    tdm_pk = join_path(tdm_t[0], tdm_t[1], '.pk')
    save_object(df, tdm_pk)
    print("Predictor type: {0}".format(type(df)))
    print("Predictor: {0} -> {1}".format(tdm_csv, tdm_pk))

    y_pk = join_path(y_t[0], y_t[1], '.pk')
    save_object(y, y_pk)
    print("Y type: {0}".format(type(y)))
    print("Y  : {0} -> {1}".format(y_csv, y_pk))

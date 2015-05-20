__author__ = '@dominofire'

import glob
import os
import pandas as pd
import re


def merge_results(folder):
    assert os.path.exists(folder) and os.path.isdir(folder)

    csv_files = glob.glob(os.path.join(folder, '*_gridSearch.csv'))

    filenames_list = [os.path.split(x)[1] for x in csv_files]

    modelnames_list = []
    for x in filenames_list:
        res = re.findall(r'(\w+)_gridSearch.csv', x)
        if len(res) == 1:
            modelnames_list.append(res[0])

    results_list = [pd.read_csv(fp) for fp in csv_files]

    colsets_list = [set(x.columns.values) for x in results_list]

    metrics_list = ['TPR', 'TNR', 'acc', 'F1', 'kappa', 'AUC', 'cut']

    metrics_colnames = set.intersection(*colsets_list)

    metrics_filtered = list(set.intersection(metrics_colnames, set(metrics_list)))

    def apply_df(df):
        params_cols = list(set.difference(set(df.columns.values), set(metrics_filtered)))
        params_cols_colnames = list(df[params_cols].columns.values)
        df['params'] = df[params_cols].apply(lambda r: ','.join(['{}={}'.format(*x)
                                                                 for x in zip(params_cols_colnames, r)]), axis=1)
        return df.loc[:, metrics_filtered + ['params']]

    results_proc = [apply_df(df) for df in results_list]

    results_final = pd.concat(results_proc)
    out_path = os.path.join(folder, 'AllResults.csv')
    results_final.to_csv(out_path)
    print('Results saved in {}'.format(out_path))

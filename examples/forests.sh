#! /bin/bash

set -e
set -o xtrace


# An example for using doger

# Data acquisition
# Loggig to Kaggle in this URL:
# https://www.kaggle.com/c/forest-cover-type-prediction/download/train.csv.zip
# You must upload this file to the workflow
unzip train.csv.zip
mv train.csv labeled.csv

# Split
doger split_traintest labeled.csv 0.70
doger split_xy labeled_train.csv Cover_Type
doger split_xy labeled_test.csv Cover_Type

# Convert to pickles
doger csv2pk labeled_train_predictors.csv labeled_train_response.csv
doger csv2pk labeled_test_predictors.csv labeled_test_response.csv

# Parallelization
# grid search
doger gridsearch \
    labeled_train_predictors.pk labeled_train_response.pk \
    labeled_test_predictors.pk labeled_test_response.pk \
    gridsearch_xt_config.py \
    obj out

doger gridsearch \
    labeled_train_predictors.pk labeled_train_response.pk \
    labeled_test_predictors.pk labeled_test_response.pk \
    gridsearch_rf_config.py \
    obj out

doger gridsearch \
    labeled_train_predictors.pk labeled_train_response.pk \
    labeled_test_predictors.pk labeled_test_response.pk \
    gridsearch_gnb_config.py \
    obj out

doger gridsearch \
    labeled_train_predictors.pk labeled_train_response.pk \
    labeled_test_predictors.pk labeled_test_response.pk \
    gridsearch_bnb_config.py \
    obj out

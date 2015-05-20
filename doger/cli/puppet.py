__author__ = '@dominofire'

import click
import doger.utils as utils
import doger.train_factory as train_factory
import doger.csv2pk as csv2pk_converter
import doger.split_predictor_response as split_predictor_response
import doger.split_train_test as split_train_test
import numpy as np
import doger.merge_results as merger
import os


@click.group()
def cli():
    """
    Tools for grid search in machine learning classification tasks
    """
    click.echo('doger - Grid search executor')
    click.echo('Working dir: {}'.format(os.getcwd()))


@cli.command()
@click.argument('x_train', type=click.Path(exists=True))
@click.argument('y_train', type=click.Path(exists=True))
@click.argument('x_test', type=click.Path(exists=True))
@click.argument('y_test', type=click.Path(exists=True))
@click.argument('config_dict', type=click.Path(exists=True))
@click.argument('obj_folder', type=click.Path(exists=False), default='./obj_{}'.format(train_factory.fixed_timestamp()))
@click.argument('out_folder', type=click.Path(exists=False), default='./out_{}'.format(train_factory.fixed_timestamp()))
def gridsearch(x_train, y_train, x_test, y_test, config_dict, out_folder, obj_folder):
    """
    Run grid search over Training and testing data
    """
    x_train_obj = utils.load_object(x_train)
    y_train_obj = utils.load_object(y_train)
    x_test_obj = utils.load_object(x_test)
    y_test_obj = utils.load_object(y_test)
    config_sets = eval(open(config_dict, 'r').read())

    train_factory.out_folder = out_folder
    train_factory.obj_folder = obj_folder
    train_factory.init()

    for cs in config_sets:
        print(cs['classifier_class'])
        cut_list = cs['cut_list']
        grid_search_params = cs['params_grid_search']
        classifier_name = cs['classifier_class']
        class_ref = utils.get_class(classifier_name)
        train_factory.grid_search(x_train_obj, y_train_obj, x_test_obj, y_test_obj, class_ref, cut_list, **grid_search_params)


@cli.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.argument('response_cols', nargs=-1)
def split_xy(csv_path, response_cols):
    """
    Splits CSV in Predictor (X) and response (y) separate files
    """
    rcols = list(response_cols)
    split_predictor_response.split_predictor_response(csv_path, rcols)


@cli.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.argument('train_test_ratio', type=click.FLOAT)
def split_traintest(csv_path, train_test_ratio):
    """
    Splits CSV in train and test files
    """
    if not 0. <= train_test_ratio <= 1.:
        click.echo('Invalid train/test ratio. It must be in [0,1] range')
        exit(1)
    split_train_test.split_train_test(csv_path, train_test_ratio)


@cli.command()
@click.argument('predictors_csv', type=click.Path(exists=True))
@click.argument('response_csv', type=click.Path(exists=True))
def csv2pk(predictors_csv, response_csv):
    """
    Converts predictor and response CSV files to Pickle files
    :param predictors_csv: Path to predictor CSV file
    :param response_csv: Path to CSV file that contains response variable
    :return: None
    """
    csv2pk_converter.convert_csv2pk(predictors_csv, response_csv)

@cli.command()
@click.argument('folder_path', type=click.Path(exists=True, file_okay=False))
def merge(folder_path):
    """
    Merge results of grid search
    :param folder_path:
    :return:
    """
    merger.merge_results(folder_path)

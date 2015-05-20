# -*- coding: UTF-8 -*-

__author__ = '@dominofire'

from datetime import datetime
from sklearn import metrics
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.ioff()
import os
import pandas as pd
import timeit
import doger.utils as utils
#Impresion a 2 decimales
from joblib import Parallel, delayed


puppet_init_time = ''


def fixed_timestamp():
    global puppet_init_time
    if puppet_init_time == '':
        puppet_init_time = str(datetime.today().strftime("%Y-%m-%d %H.%M.%s"))
    return puppet_init_time


out_folder = './out_{0}/'.format(fixed_timestamp())
obj_folder = './obj_{0}/'.format(fixed_timestamp())


class PrettyFloat(float):
    def __repr__(self):
        return '{0:.2f}'.format(self)


def pretty_floats(obj):
    if isinstance(obj, float):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return map(pretty_floats, obj)
    return obj


def calc_perc_cm(cm):
    """
    A partir de una matriz de confusión calcula una nueva CM con valores de porcentaje por filas
    :param cm: Matriz de confusión total
    return: cm2: Matriz de confusión en porcentaje
    """
    cm2 = np.zeros(np.shape(cm), dtype=float)
    for i in range((np.shape(cm)[0])):
        for j in range((np.shape(cm)[1])):
            if cm[i, j] != 0:
                cm2[i, j] = float(cm[i, j])/sum(cm[i])
    return cm2


def plot_cm(cm1, pathout, name, labels):
    """
    Imprime matrices de confusión (total y porcentaje por fila)
    :param cm1: Matriz de confusion total
    :param name:
    :param date:
    :param labels; Etiquetas a poner en las matrices de confusion
    :return:
    """
    date = str(datetime.today().strftime("%Y-%m-%d %H.%M.%s"))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm1)
    plt.title('Confusion matrix '+str(name))

    #numeritos
    for i, cas in enumerate(cm1):
        for j, c in enumerate(cas):
            if c > 0:
                plt.text(j-.2, i+.2, '{:.2f}'.format(c), fontsize=14)
    fig.colorbar(cax)

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(pathout, '{} CM {}.pdf'.format(name, date)), format='pdf')
    plt.close()


def init():
    print('Creating out and obj folders')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(obj_folder):
        os.makedirs(obj_folder)


def get_classifier_name(estimator, **params):
    return estimator.__class__.__name__ + '~' + '~'.join(['{0}={1}'.format(n, v) for n, v in params.items()])


def compute_sensitivity(cm):
    assert len(cm.shape) and cm.shape[0] == cm.shape[1] and cm.shape[0] == 2
    TP = cm[1, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    if TP+FN == 0:
        return 0
    return float(TP)/(TP + FN)


def compute_specificity(cm):
    """
    Computes specificity for binary classification problems
    :param cm: A Numpy-like matrix that represents a confusion matrix
    :return: The specificity of the confusion matrix
    """
    print(cm.shape)
    print(len(cm.shape))
    assert len(cm.shape) and cm.shape[0] == cm.shape[1] and cm.shape[0] == 2
    TP = cm[1, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    if FP+TN == 0:
        return 0
    return float(TN)/(FP + TN)


def plot_ROC(y_test, y_proba, folder, name):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=1)
    auc_val = auc(fpr, tpr)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line_points = np.arange(0, 1.05, 0.05)
    ax1.plot(fpr, tpr, 'g-', line_points, line_points, 'r-')

    plt.title('ROC {0}'.format(name))
    plt.savefig(os.path.join(folder, '{0} ROC {1}.pdf'.format(name, puppet_init_time)), format='pdf')
    plt.close()
    return auc_val


def compute_kappa(cm):
    import numpy as np
    tot = np.sum(cm)
    a = np.trace(cm) #si se divide entre tot es el prcentaje de agreement
    ef_vect = np.zeros(len(cm))
    for i in range(0, len(cm)):
        ef_vect[i] = np.sum(cm[i, :]) * np.sum(cm[:, i]) / tot
    ef = np.sum(ef_vect)
    return (a-ef)/(tot-ef)


def compute_F1(cm):
    """
    For Binary classification problems only
    :param cm:
    :return:
    """
    assert len(cm.shape) and cm.shape[0] == cm.shape[1] and cm.shape[0] == 2
    TP = cm[1, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    assert (2*TP + FP + FN) != 0
    return float(2*TP)/(2*TP + FP + FN)


def predict_classes(probas, cut):
    """
    Predice si la probabilidad de ser malo cae dentro del rango del corte
    :probas: numpy array or list
    """
    N = len(probas)
    res = np.zeros(N)
    for i in range(N):
        if probas[i] < cut:
            res[i] = 0
        else:
            res[i] = 1
    return res


def train_classifier(X, y, classifier_class, **parameters):
    print(classifier_class)
    print(parameters)
    t_start = timeit.default_timer()
    estimator = classifier_class()
    estimator.set_params(**parameters)
    estimator.fit(X, y)
    duration = timeit.default_timer() - t_start
    utils.save_object(estimator,
        os.path.join(obj_folder, '{0}.pk'.format(get_classifier_name(estimator, **parameters)))
    )
    return estimator, duration


def test_classifier(X_test, y_test, cl, cut, basename):
    """

    :param X_test:
    :param y_test:
    :param cl:
    :return:
    """
    unique_labels = np.lib.unique(y_test)
    labels = np.append('', unique_labels)
    time_mark = timeit.default_timer()

    if len(unique_labels) > 2:
        y_pred = cl.predict(X_test)
    else:
        y_proba = cl.predict_proba(X_test)
        y_pred = predict_classes(y_proba[:, 1], cut)

    acc = accuracy_score(y_test, y_pred, normalize=True)

    cm = confusion_matrix(y_test, y_pred)
    cm_perc = calc_perc_cm(cm)
    #tnr = compute_specificity(cm)
    #tpr = compute_sensitivity(cm)
    kappa = compute_kappa(cm)
    #f1 = compute_F1(cm)

    plot_cm(cm, out_folder, '{0}$cut={1:.4f} (Total)'.format(basename, cut), labels)
    plot_cm(cm_perc, out_folder, '{0}$cut={1:.4f} (Perc)'.format(basename, cut), labels)
    #auc_m = plot_ROC(y_test, y_proba[:, 1], out_folder, basename)
    #auc_m = -1
    #performance reports
    fout = open(os.path.join(out_folder, '{0}$cut={1:.4f} report.txt'.format(basename, cut)), 'w')
    fout.write(metrics.classification_report(y_test, y_pred))
    fout.close()

    time_mark = time_mark - timeit.default_timer()
    #return {'CM': cm, 'acc': acc, 'kappa': kappa, 'TPR': tpr, 'TNR': tnr, 'AUC': auc_m, 'duration': time_mark, 'F1': f1}

    return {'CM': cm, 'acc': acc, 'kappa': kappa, 'duration': time_mark}


def test_cut(params, X_test, y_test, est, c, basename):
    pars = dict(params)
    res = test_classifier(X_test, y_test, est, c, basename)
    pars['acc'] = res['acc']
    pars['kappa'] = res['kappa']
    #pars['TPR'] = res['TPR']
    #pars['TNR'] = res['TNR']
    #pars['AUC'] = res['AUC']
    #pars['F1'] = res['F1']
    pars['cut'] = c
    #print pretty_floats(pars)
    return pars


def train_test_cut(params, X_train, y_train, classifier_class, X_test, y_test, cut_list):
    est, d1 = train_classifier(X_train, y_train, classifier_class, **params)
    basename = get_classifier_name(est, **params)
    results = [test_cut(params, X_test, y_test, est, c, basename) for c in cut_list]
    return results


def grid_search(X_train, y_train, X_test, y_test, classifier_class, cut_list, **params_grid):
    params_col = list(ParameterGrid(params_grid))
    results = []
    out_filename = os.path.join(out_folder, '{0}_gridSearch.csv'.format(classifier_class.__name__))

    res = Parallel(n_jobs=1)(delayed(train_test_cut)
                             (pars, X_train, y_train, classifier_class, X_test, y_test, cut_list)
                             for pars in params_col)
    # Flatten
    for subr in res:
        results.extend(subr)

    pd.DataFrame(results).to_csv(out_filename, index=False)
    print('Results saved in {}'.format(out_filename))

    return results

# -*- coding: UTF-8 -*-

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy.sparse import csr_matrix


def load_object(filename):
    """
    Loads a Pickle-serialized object stored in file pointed by filename path
    """
    with open(filename, 'rb') as inp:
        h = pickle.load(inp)
    return h


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
 

def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m


# def make_word_cloud(df, k, minsize, maxsize, pathout, name):
#     """
#     Crea un word cloud
#     :param date: Fecha
#     :param df: Pandas dataframe con formato (palabra, count)
#     :k: Número de palabras en el wordcloud (130)
#     :minsize: tamaño mínimo de cada palabra (20)
#     :maxsize: tamaño máximo de cada palabra (80)
#     :name: nombre del clasificador
#     """
#     from pytagcloud import create_tag_image, make_tags
#     print 'Hacer wordcloud'
#     date = str(datetime.today().strftime("%Y-%m-%d %H.%M.%s"))
#     tup = df.order(ascending=False)[0:k].to_dict()
#     tup = list(tup.items())
#     tags = make_tags(tup, maxsize=maxsize, minsize=minsize)
#     create_tag_image(tags, pathout+'Wordcloud '+name+' '+date+'.png', size=(900, 600))


def split_path(file_location):
    """
    Divide un path de noticias en una tupla de la forma (carpeta, nombre, extension)
    """
    file_path, file_name = os.path.split(file_location)
    file_base, file_ext = os.path.splitext(file_name)
    return file_path, file_base, file_ext


def join_path(tuple_path):
    """
    Crea una ruta valida desde una tupla de la forma (carpeta, nombre, extension)
    """
    return os.path.join(tuple_path[1], tuple_path[1] + tuple_path[2])


def join_path(base, name, ext):
    """
    Crea una ruta valida desde los parametros base/name.ext
    """
    return os.path.join(base, name + ext)


def to_tdm_df(np_matrix, global_dict_list):
    if not (isinstance(np_matrix, pd.DataFrame) or isinstance(np_matrix, csr_matrix)):
        raise ValueError('np_matrix must be pd.DataFrame or sp.csr_matrix')
    if not isinstance(global_dict_list, list):
        raise ValueError('global_dict_list must be a list')
    if isinstance(np_matrix, pd.DataFrame):
        return pd.DataFrame(np_matrix, columns=global_dict_list)
    elif isinstance(np_matrix, csr_matrix):
        return pd.DataFrame(data=np.array(np_matrix.todense()), columns=global_dict_list)

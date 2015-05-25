# coding: utf-8
from __future__ import division, print_function, unicode_literals
import pandas as pd

def contains_na_columns(d):
    """
    欠損値を含む列のpandas.Indexを返す

    Parameters
    ----------
    d : pandas.DataFrame
    """
    return d.columns.difference(d.dropna(axis=1).columns)

# ----------------
# データの読み込み
# ----------------
d = pd.read_csv(
    'adult.data', sep=', ',
    names=('age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'label'),
    engine='python',
    na_values='?')

# ------------
# データの加工
# ------------
# In [ ]: contains_na_columns(d)
# Out[ ]: Index([u'native_country', u'occupation', u'workclass'], dtype='object')
d['label'] = (d['label'] == '>50K')

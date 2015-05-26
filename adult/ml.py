# coding: utf-8
from __future__ import division, print_function, unicode_literals
import numpy
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn_pandas import DataFrameMapper, RandomizedSearchCV

def contains_na_columns(d):
    """
    欠損値を含む列のpandas.Indexを返す

    Parameters
    ----------
    d : pandas.DataFrame
    """
    return d.columns.difference(d.dropna(axis=1).columns)

def best_cv_num(n):
    return int(1+numpy.log2(n))

#
# データの読み込み
#
d = pd.read_csv(
    'adult.data', sep=', ',
    names=('age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'label'),
    engine='python',
    na_values='?')

#
# データの加工
#
# In [ ]: contains_na_columns(d)
# Out[ ]: Index([u'native_country', u'occupation', u'workclass'], dtype='object')
d.loc[:,'label'] = (d.loc[:,'label'] == '>50K')
_cols = contains_na_columns(d)
d.loc[:,_cols] = d.loc[:,_cols].fillna('?')

mapper = DataFrameMapper([
    ('age', StandardScaler()),
    ('workclass', LabelBinarizer()),
    ('fnlwgt', StandardScaler()),
    ('education', LabelBinarizer()),
    ('education_num', StandardScaler()),
    ('marital_status', LabelBinarizer()),
    ('occupation', LabelBinarizer()),
    ('relationship', LabelBinarizer()),
    ('race', LabelBinarizer()),
    ('sex', LabelBinarizer()),
    ('capital_gain', StandardScaler()),
    ('capital_loss', StandardScaler()),
    ('hours_per_week', StandardScaler()),
    ('native_country', LabelBinarizer()),
    ])
p = Pipeline([
    ('mapper', mapper),
    ('clf', LinearSVC()),
    ])
_n = d.shape[0]
cv = RandomizedSearchCV(p, {'clf__C': 2**numpy.linspace(-5,3,1000)}, n_iter=10, cv=best_cv_num(_n), n_jobs=-1, verbose=3)
cv.fit(d, d.loc[:,'label'].values)

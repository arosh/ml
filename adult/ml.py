# coding: utf-8
from __future__ import division, print_function, unicode_literals
import numpy
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.feature_extraction import DictVectorizer
from scipy.stats import randint as sp_randint
from operator import itemgetter

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


if __name__ == '__main__':
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
    _cols = ['native_country', 'occupation', 'workclass']
    d.loc[:,_cols] = d.loc[:,_cols].fillna('?')

    X = d.loc[:, d.columns != 'label'].T.to_dict().values()
    X = DictVectorizer().fit_transform(X)
    X = StandardScaler(with_mean=False).fit_transform(X)
    y = d.loc[:, 'label'].values
    _n = d.shape[0]

    # http://scikit-learn.org/stable/modules/feature_selection.html#selecting-non-zero-coefficients
    p = Pipeline([
        ('clf', LinearSVC(penalty='l1', dual=False)),
        ])

    params = {
        'clf__C': 2**numpy.linspace(-3,3,1000),
    }
    cv = RandomizedSearchCV(p, params, n_iter=20, cv=best_cv_num(_n), n_jobs=2, verbose=3)
    cv.fit(X, y)
    print(cv.best_score_)
    print(cv.best_params_)
    """
    factor = cv.best_estimator_.named_steps['clf'].coef_.tolist()
    result = zip(factor, categorical_columns(d))
    result.sort(key=itemgetter(0), reverse=True)
    for a,b in result[:20]:
            print(a,b)
    """
    
    """
    p = RandomForestClassifier()
    params = {
            "max_depth": [3, None],
            "max_features": sp_randint(1, 11),
            "min_samples_split": sp_randint(1, 11),
            "min_samples_leaf": sp_randint(1, 11),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
    }
    cv = RandomizedSearchCV(p, params, n_iter=100, cv=best_cv_num(_n), n_jobs=-1, verbose=1)
    cv.fit(X, y)
    print(cv.best_score_)
    print(cv.best_params_)
    """

    """
    clf = GradientBoostingClassifier()
    cv = RandomizedSearchCV(clf, {}, n_iter=1, cv=best_cv_num(_n), n_jobs=-1, verbose=3)
    cv.fit(X, y)
    print(cv.best_score_)
    #cv = clf
    #cv.fit(X, y)
    factor = cv.best_estimator_.feature_importances_.tolist()
    result = zip(factor, categorical_columns(d))
    result.sort(key=itemgetter(0), reverse=True)
    for a,b in result[:20]:
            print(a,b)
    """

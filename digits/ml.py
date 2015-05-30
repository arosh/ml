from __future__ import division, print_function, unicode_literals
import numpy
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV

def best_cv_num(n):
    return int(1+numpy.log2(n))

def best_n_iter(n):
    return numpy.ceil(10**6 / n)

if __name__ == '__main__':
    d = load_digits()
    X = d.data
    X = StandardScaler().fit_transform(X)
    y = d.target
    _n = X.shape[0]

    # http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    clf = SVC()
    params = {
            'C': 2**numpy.linspace(-5,15),
            'gamma': 2**numpy.linspace(-15,3),
            'class_weight': [None, 'auto'],
            }
    cv = RandomizedSearchCV(clf, params, n_iter=20, cv=best_cv_num(_n), n_jobs=2, verbose=1)
    cv.fit(X, y)
    print(cv.best_score_)
    print(cv.best_params_)

from __future__ import division, print_function, unicode_literals
import numpy
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.grid_search import RandomizedSearchCV

def best_cv_num(n):
    return int(1+numpy.log2(n))

def best_n_iter(n):
    return numpy.ceil(10**6 / n)

if __name__ == '__main__':
    d = fetch_20newsgroups_vectorized(
            remove=('headers', 'footers', 'quotes'))
    X = d.data
    #X = StandardScaler(with_mean=False).fit_transform(X)
    #X = TruncatedSVD(n_components=400).fit_transform(X)
    y = d.target
    _n = X.shape[0]

    clf = MultinomialNB()
    params = {
            'alpha': numpy.linspace(0,1,1000)
    }

    # http://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use
    """
    clf = SGDClassifier(n_iter=best_n_iter(_n))
    params = {
            'alpha': 10**numpy.linspace(-7,-1,1000),
    }
    """
    cv = RandomizedSearchCV(clf, params, n_iter=20, cv=best_cv_num(_n), n_jobs=2, verbose=3)
    cv.fit(X, y)
    print(cv.best_score_)
    print(cv.best_params_)

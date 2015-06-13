# coding: utf_8
from __future__ import division, print_function, unicode_literals
from future_builtins import *

import numpy
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.cm
from chainer import Variable, FunctionSet, optimizers
import chainer.functions as F


def best_cv_num(n):
    return int(1+numpy.log2(n))


class DnnRegressor(object):
    def __init__(self, model, forward, opt=optimizers.Adam()):
        self.model = model
        self.forward = forward
        self.opt = opt
        self.opt.setup(model.collect_parameters())

    def train(self, X, y, batch_size, n_epoch=20, cv=None, verbose=True):
        from sklearn.utils import shuffle

        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        n_samples, n_features = X.shape

        if y.ndim == 1:
            y.shape = (-1, 1)

        if cv is None:
            cv = best_cv_num(n_samples)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 / cv)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        self.train_error = []
        self.test_error = []

        for epoch in xrange(1, n_epoch+1):
            if verbose:
                print('epoch = {}'.format(epoch))

            # train
            X_train, y_train = shuffle(X_train, y_train)
            for i in xrange(0, n_train, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                self.opt.zero_grads()
                error = forward(X_batch, y_batch, train=True)
                error.backward()
                self.opt.update()

            error = forward(X_train, y_train)
            self.train_error.append(float(error.data))

            # test
            error = forward(X_test, y_test)
            self.test_error.append(float(error.data))

            if verbose:
                print('train error={}'.format(self.train_error[-1]))
                print('test  error={}'.format(self.test_error[-1]))


if __name__ == '__main__':
    d = load_boston()
    X = d.data
    X = scale(X)
    y = d.target
    n_samples, n_features = X.shape

    # http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    # from sklearn.svm import SVR
    # clf = SVR()
    # params = {
    #         'C': 2**numpy.linspace(-5,15),
    #         'gamma': 2**numpy.linspace(-15,3),
    #         }

    # from sklearn.linear_model import SGDRegressor
    # def best_n_iter(n):
    #     return numpy.ceil(10**6 / n)
    # clf = SGDRegressor(n_iter=best_n_iter(n_samples))
    # params = {
    #         'alpha': 10**numpy.linspace(-7,-1,1000),
    # }

    # from sklearn.grid_search import RandomizedSearchCV
    # cv = RandomizedSearchCV(clf, params, scoring='mean_squared_error', n_iter=50, cv=best_cv_num(n_samples), n_jobs=2, verbose=1)
    # cv.fit(X, y)
    # print(cv.best_score_)
    # print(cv.best_params_)

    n_units = 50
    batch_size = 10

    model = FunctionSet(l1=F.Linear(n_features, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, 1))

    def forward(X_data, y_data, train=False):
        X, t = Variable(X_data), Variable(y_data)
        h1 = F.dropout(F.relu(model.l1(X)),  train=train)
        h2 = F.dropout(F.relu(model.l2(h1)), train=train)
        y = model.l3(h2)
        return F.mean_squared_error(y, t)

    clf = DnnRegressor(model, forward)
    clf.train(X, y, batch_size, n_epoch=50)

    colors = matplotlib.cm.rainbow(numpy.linspace(0, 1, 2))
    plt.scatter(numpy.arange(1,len(clf.train_error)+1), clf.train_error, color=colors[0], label='train')
    plt.scatter(numpy.arange(1,len(clf.test_error)+1), clf.test_error, color=colors[1], label='test')
    plt.legend(loc='best')
    plt.show()

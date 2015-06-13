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
    # def best_n_iter(n):
    #     return numpy.ceil(10**6 / n)

    # from sklearn.linear_model import SGDRegressor
    # clf = SGDRegressor(n_iter=best_n_iter(n_samples))
    # params = {
    #         'alpha': 10**numpy.linspace(-7,-1,1000),
    # }
    # from sklearn.grid_search import RandomizedSearchCV
    # cv = RandomizedSearchCV(clf, params, scoring='mean_squared_error', n_iter=50, cv=best_cv_num(n_samples), n_jobs=2, verbose=1)
    # cv.fit(X, y)
    # print(cv.best_score_)
    # print(cv.best_params_)

    test_size = 1 / best_cv_num(n_samples)
    n_epoch = 20
    n_units = 50
    batch_size = 10

    X = X.astype(numpy.float32)
    y = y.astype(numpy.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    model = FunctionSet(l1=F.Linear(n_features, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, 1))

    def forward(X_data, y_data, train=False):
        X, t = Variable(X_data), Variable(y_data)
        h1 = F.dropout(F.relu(model.l1(X)),  train=train)
        h2 = F.dropout(F.relu(model.l2(h1)), train=train)
        y = model.l3(h2)
        return F.mean_squared_error(y, t)

    opt = optimizers.Adam()
    opt.setup(model.collect_parameters())

    train_error = []
    test_error = []

    for epoch in xrange(1, n_epoch+1):
        print('epoch = {}'.format(epoch))

        # train
        X_train, y_train = shuffle(X_train, y_train)
        for i in xrange(0, n_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            opt.zero_grads()
            error = forward(X_batch, y_batch, train=True)
            error.backward()
            opt.update()

        error = forward(X_train, y_train)
        train_error.append(float(error.data))

        # test
        X_test, y_test = shuffle(X_test, y_test)
        error = forward(X_test, y_test)
        test_error.append(float(error.data))

        print('MSE train={} test={}'.format(train_error[-1], test_error[-1]))

    colors = matplotlib.cm.rainbow(numpy.linspace(0, 1, 2))
    plt.scatter(numpy.arange(1,n_epoch+1), train_error, color=colors[0], label='train')
    plt.scatter(numpy.arange(1,n_epoch+1), test_error, color=colors[1], label='test')
    plt.legend(loc='best')
    plt.show()

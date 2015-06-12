# coding: utf_8
from __future__ import division, print_function, unicode_literals
from future_builtins import *

import numpy
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm
from chainer import Variable, FunctionSet, optimizers
import chainer.functions as F

def best_cv_num(n):
    return int(1+numpy.log2(n))

if __name__ == '__main__':
    d = load_boston()
    X = d.data
    X = StandardScaler().fit_transform(X)
    X = X.astype(numpy.float32)
    y = d.target
    y = y.astype(numpy.float32)
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

    batchsize = 100
    n_epoch = 50
    n_units = 100
    test_size = 1 / best_cv_num(n_samples)
    model = FunctionSet(l1=F.Linear(n_features, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, 1))

    def forward(X_data, y_data, train=False):
        X, t = Variable(X_data), Variable(y_data)
        h1 = F.dropout(F.relu(model.l1(X)), train=train)
        h2 = F.dropout(F.relu(model.l2(h1)), train=train)
        y = model.l3(h2)
        return F.mean_squared_error(y, t)

    opt = optimizers.Adam()
    opt.setup(model.collect_parameters())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    print('n_samples={}'.format(n_samples))
    print('n_train={}, n_test={}'.format(n_train, n_test))

    error_train = []
    error_test = []

    for epoch in xrange(n_epoch):
        print('epoch = {}'.format(epoch))

        # train
        sum_error = 0
        for i in xrange(0, n_train, batchsize):
            X_batch = X_train[i:i+batchsize,:]
            y_batch = y_train[i:i+batchsize,:]

            opt.zero_grads()
            error = forward(X_batch, y_batch, train=True)
            error.backward()
            opt.update()

            sum_error += error.data * X_batch.shape[0]
        error_train.append(sum_error / n_train)

        # test
        sum_error = 0
        for i in xrange(0, n_test, batchsize):
            X_batch = X_test[i:i+batchsize,:]
            y_batch = y_test[i:i+batchsize,:]

            error = forward(X_test, y_test)
            sum_error += error.data * X_batch.shape[0]
        error_test.append(sum_error / n_test)
        print('MSE = {}'.format(sum_error / n_test))

    colors = matplotlib.cm.rainbow(numpy.linspace(0, 1, 2))
    plt.scatter(numpy.arange(1,n_epoch+1), error_train, color=colors[0], label='train')
    plt.scatter(numpy.arange(1,n_epoch+1), error_test, color=colors[1], label='test')
    plt.legend(loc='best')
    plt.show()

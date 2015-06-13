# coding: utf_8
from __future__ import division, print_function, unicode_literals
from future_builtins import *

import numpy

from sklearn.datasets import make_blobs
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from chainer import Variable, FunctionSet, optimizers
import chainer.functions as F


def best_cv_num(n):
    return int(1+numpy.log2(n))


class DnnClassifier(object):
    def __init__(self, model, forward, opt=optimizers.Adam()):
        self.model = model
        self.forward = forward
        self.opt = opt
        self.opt.setup(model.collect_parameters())

    def train(self, X, y, batch_size, n_epoch=20, cv=None, verbose=True):
        from sklearn.utils import shuffle

        X = X.astype(numpy.float32)
        y = y.astype(numpy.int32)

        n_samples, n_features = X.shape

        if y.ndim != 1:
            raise 'y.dim should be 1'

        if cv is None:
            cv = best_cv_num(n_samples)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 / cv)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        for epoch in xrange(1, n_epoch+1):
            if verbose:
                print('epoch = {}'.format(epoch))

            # train
            X_train, y_train = shuffle(X_train, y_train)
            for i in xrange(0, n_train, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                self.opt.zero_grads()
                loss, _ = forward(X_batch, y_batch, train=True)
                loss.backward()
                self.opt.update()

            loss, acc = forward(X_train, y_train)
            self.train_loss.append(float(loss.data))
            self.train_accuracy.append(float(acc.data))

            # test
            loss, acc = forward(X_test, y_test)
            self.test_loss.append(float(loss.data))
            self.test_accuracy.append(float(acc.data))

            if verbose:
                print('train loss={} accuracy={}'.format(self.train_loss[-1], self.train_accuracy[-1]))
                print('test  loss={} accuracy={}'.format(self.test_loss[-1], self.test_accuracy[-1]))


if __name__ == '__main__':
    X, y = make_blobs(
        n_samples=10000,
        n_features=2,
        centers=[[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]],
        cluster_std=0.2)
    y[y == 2] = 0
    y[y == 3] = 1
    n_samples, n_features = X.shape

    # print('n_samples =', n_samples)
    # print('n_features =', n_features)
    # print('n_train =', n_train)
    # print('n_test =', n_test)
    # print('X_train.shape =', X_train.shape)
    # print('y_train.shape =', y_train.shape)
    # print('X_test.shape =', X_test.shape)
    # print('y_test.shape =', y_test.shape)

    # num_colors = 2
    # colors = cm.rainbow(numpy.linspace(0, 1, num_colors))
    # plt.scatter(X[:, 0], X[:, 1], color=colors[y[:,0]])
    # plt.show()

    batch_size = 50
    n_units = 12

    model = FunctionSet(l1=F.Linear(n_features, n_units),
                        l2=F.Linear(n_units, 2))

    def forward(X_data, y_data, train=False):
        x = Variable(X_data)
        t = Variable(y_data)
        h1 = F.dropout(F.relu(model.l1(x)), train=train)
        y  = model.l2(h1)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    clf = DnnClassifier(model, forward)
    clf.train(X, y, batch_size)

    colors = cm.rainbow(numpy.linspace(0, 1, num=2))
    plt.scatter(numpy.arange(1,len(clf.train_accuracy)+1), clf.train_accuracy, label='train', color=colors[0])
    plt.scatter(numpy.arange(1,len(clf.test_accuracy)+1), clf.test_accuracy, label='test', color=colors[1])
    plt.legend(loc='best')
    plt.show()

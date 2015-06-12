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


if __name__ == '__main__':
    X, y = make_blobs(
        n_samples=10000,
        n_features=2,
        centers=[[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]],
        cluster_std=0.2)
    y[y == 2] = 0
    y[y == 3] = 1

    n_samples, n_features = X.shape

    test_size = 1 / best_cv_num(n_samples)
    n_epoch = 20
    n_units = 12
    batch_size = 50

    X = X.astype(numpy.float32)
    y = y.astype(numpy.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

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

    model = FunctionSet(l1=F.Linear(n_features, n_units),
                        l2=F.Linear(n_units, 2))

    def forward(X_data, y_data, train=False):
        x = Variable(X_data)
        t = Variable(y_data)
        h1 = F.dropout(F.relu(model.l1(x)), train=train)
        y  = model.l2(h1)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    opt = optimizers.Adam()
    opt.setup(model.collect_parameters())

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    for epoch in xrange(1, n_epoch+1):
        print('epoch = {}'.format(epoch))

        # train
        sum_loss = 0
        sum_accuracy = 0
        X_train, y_train = shuffle(X_train, y_train)
        for i in xrange(0, n_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            n_batch = X_batch.shape[0]

            opt.zero_grads()
            loss, _ = forward(X_batch, y_batch, train=True)
            loss.backward()
            opt.update()

            loss, acc = forward(X_batch, y_batch)
            sum_loss     += float(loss.data) * n_batch
            sum_accuracy += float(acc.data)  * n_batch

        train_loss.append(sum_loss / n_train)
        train_accuracy.append(sum_accuracy / n_train)

        # test
        X_test, y_test = shuffle(X_test, y_test)

        loss, acc = forward(X_test, y_test)
        sum_loss     = float(loss.data)
        sum_accuracy = float(acc.data)

        test_loss.append(sum_loss)
        test_accuracy.append(sum_accuracy)

        print('train loss={} accuracy={}'.format(train_loss[-1], train_accuracy[-1]))
        print('test  loss={} accuracy={}'.format(test_loss[-1], test_accuracy[-1]))

    colors = cm.rainbow(numpy.linspace(0, 1, num=2))
    plt.scatter(numpy.arange(1,n_epoch+1), train_accuracy, label='train', color=colors[0])
    plt.scatter(numpy.arange(1,n_epoch+1), test_accuracy, label='test', color=colors[1])
    plt.legend(loc='best')
    plt.show()

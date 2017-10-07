"""
wn algorithm for testing
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
import numpy as np
from scipy import sparse

from surprise import AlgoBase
from surprise import Dataset
from surprise import evaluate


class MyOwnAlgorithm(AlgoBase):

    def __init__(self, factor_n, epoch_n, learning_rate=0.01, reg=0.1):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.n_factor = factor_n
        self.epoch_n = epoch_n
        self.learning_rate = learning_rate
        self.reg = reg
        self.rating = None
        self.mu = None
        self.bu = None
        self.bi = None
        self.p = None
        self.q = None

    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        # Compute the average rating. We might as well use the
        # trainset.global_mean attribute ;)
        num_user = trainset.n_users
        num_item = trainset.n_items
        mu = trainset.global_mean
        bu = np.matrix(np.zeros([num_user, 1]))
        bi = np.matrix(np.zeros([num_item, 1]))
        p = np.matrix(np.zeros([self.n_factor, num_user])) + 0.001
        q = np.matrix(np.zeros([self.n_factor, num_item])) + 0.001
        rating = sparse.lil_matrix((num_user, num_item))

        for u, i, r in trainset.all_ratings():
            rating[u, i] = r

        rating = sparse.dok_matrix(rating)
        

        for f in range(self.n_factor):
            for i_epoch in range(self.epoch_n):
                for k, v in rating.items():
                    # k[0]: user
                    # k[1]: item
                    err = v - mu - bu[k[0]] - bi[k[1]] - \
                        q[:f + 1, k[1]].T * p[:f + 1, k[0]]
                    if math.isnan(err):
                        print(bu)
                        print(bi)
                        print(p)
                        print(q)
                        continue
                    bu[k[0]] += self.learning_rate * \
                        (err - self.reg * bu[k[0]])
                    bi[k[1]] += self.learning_rate * \
                        (err - self.reg * bi[k[1]])
                    q[f, k[1]] += self.learning_rate * \
                        (err * p[f, k[0]] + self.reg * q[f, k[1]])
                    p[f, k[0]] += self.learning_rate * \
                        (err * q[f, k[1]] + self.reg * q[f, k[0]])
        

        self.mu = mu
        self.bu = bu
        self.bi = bi
        self.p = p
        self.q = q

    def estimate(self, u, i):

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        est = self.trainset.global_mean

        if known_user:
            est += self.bu[u]

        if known_item:
            est += self.bi[i]

        if known_user and known_item:
            est += np.dot(self.q[:, i].T, self.p[:, u])

        else:
            return 3
            raise PredictionImpossible('User and item are unkown.')

        return est


data = Dataset.load_builtin('ml-100k')

algo = MyOwnAlgorithm(10,10)
evaluate(algo, data)

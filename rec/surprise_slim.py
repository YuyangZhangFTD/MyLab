"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy import sparse

from surprise import AlgoBase
from surprise import Dataset
from surprise import evaluate
import slim_tool as st


class MyOwnAlgorithm(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

        self.W = None
        self.A = None

    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        # Compute the average rating. We might as well use the
        # trainset.global_mean attribute ;)
        # self.the_mean = np.mean([r for (_, _, r) in
        #                          self.trainset.all_ratings()])

        user_num = trainset.n_users
        item_num = trainset.n_items
        rating = sparse.lil_matrix((user_num, item_num))

        print("rating shape:  " + str(rating.shape))

        for u, i, r in trainset.all_ratings():
            rating[u, i] = r

        self.A = st.SparseMatrix((item_num, item_num))

        self.W = st.train(
            self.A,
            l1_ratio=0.1,
            eps=1e-4,
            n_alphas=100,
            max_iter=100)

        self.A = sparse.csc_matrix(self.A)
        self.W = sparse.csc_matrix(self.W)

        self.estimator = self.A * self.W

        print(self.estimator.shape)

        print(self.estimator.toarray())

    def estimate(self, u, i):

        try:
            return self.estimator[u, i]
        except BaseException:
            print("error at: u->" + str(u) + "  i->" + str(i))
            return 3


data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

evaluate(algo, data)

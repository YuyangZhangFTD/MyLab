"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from scipy import sparse
from surprise import AlgoBase
from surprise import Dataset

import slim_tool as st
from MySurpriseEnv import *


class MyOwnAlgorithm(AlgoBase):

    def __init__(
            self,
            l1_ratio=0.1,
            eps=1e-3,
            n_alphas=10,
            alphas=None,
            positive=True,
            max_iter=100):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.W = None
        self.A = None
        self.trainset = None
        self.the_mean = 0
        self.trainset = None
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.alphas = alphas
        self.positive = positive

    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        rating = sparse.lil_matrix((user_num, item_num))

        for u, i, r in self.trainset.all_ratings():
            if r > 3:
                rating[u, i] = r

        self.A = st.SparseMatrix(rating)

        self.W = st.train(
            self.A,
            l1_ratio=self.l1_ratio,
            eps=self.eps,
            n_alphas=self.n_alphas,
            alphas=self.alphas,
            positive=self.positive,
            max_iter=self.max_iter)

        self.A = sparse.csc_matrix(self.A)
        self.W = sparse.csc_matrix(self.W)

        self.estimator = self.A * self.W

    def estimate(self, u, i):

        try:
            return self.estimator[u, i]
        except BaseException:
            print("error at: u->" + str(u) + "  i->" + str(i))
            return 3

    def predict_topn(self, uid, iid, topn, verbose=False):
        """ Not implement self.estimate

        And generate recommendation list of a user here
        """
        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:

            user_rating = self.estimator[iuid, :].toarray()

            index_rating = [(i, user_rating[i])
                            for i in range(len(user_rating))]

            index_rating.sort(key=lambda x: -x[1])

            est_list = [x[0] for x in index_rating[:topn]]

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est_list = []
            details['was_impossible'] = True
            details['reason'] = str(e)

        pred = Prediction_topn(uid, iiid, est_list, details)

        if verbose:
            print(pred)

        return pred

    def test_topn(self, testset, topn, verbose=False):
        predictions_topn = [
            self.predict_topn(
                uid,
                iid,
                topn,
                verbose=verbose) for (
                uid,
                iid,
                r_ui_trans) in testset if r_ui_trans > 3]
        return predictions_topn


if __name__ == '__main__':
    data = Dataset.load_builtin('ml-100k')
    algo = MyOwnAlgorithm(
        l1_ratio=1,
        eps=1e-3,
        n_alphas=100,
        # alphas=[10],
        positive=True,
        max_iter=10000)

    # evaluate(algo, data)
    evaluate_topn(algo, data, 10)

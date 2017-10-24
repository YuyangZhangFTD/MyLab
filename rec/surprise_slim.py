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
from MySurprise import *


class MyOwnAlgorithm(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

        self.W = None
        self.A = None

    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

	# the mean value for selecting recommendation test set
	self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        # Compute the average rating. We might as well use the
        # trainset.global_mean attribute ;)
        # self.the_mean = np.mean([r for (_, _, r) in
        #                          self.trainset.all_ratings()])

        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        rating = sparse.lil_matrix((user_num, item_num))

        print("rating shape:  " + str(rating.shape))

        for u, i, r in self.trainset.all_ratings():
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
            
            user_rating = self.estimator[uid, :].toarray()

            index_rating = [i, user_rating[i] for i in range(user_rating)]

	    index_rating.sort(key=lambda x: -x[1])	    

	    est_list = [x[0] for x in index_rating[:top]]

            details['was_impossible'] = False

	except PredictionImpossible as e:
            est = []
            details['was_impossible'] = True
            details['reason'] = str(e)

	pred = Prediction_topn(uid, iid, est_list, details)

        if verbose:
            print(pred)

        return pred


    def test_topn(self, testset, topn, verbose=False):
        predictions = [self.predict_topn(uid,
                                    iid,
                                    topn,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset if r_ui_trans>self.the_mean]
        return predictions


data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

evaluate_topn(algo, data, 10)

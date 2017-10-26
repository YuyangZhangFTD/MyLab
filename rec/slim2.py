"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy import sparse
from sklearn import linear_model

from surprise import AlgoBase
from surprise import Dataset
from surprise import evaluate


class MyOwnAlgorithm(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        user_num = self.trainset.n_user
        item_num = self.trainset.n_item
        A = sparse.lil_matrix((user_num, item_num))

        the_mean = np.mean([r for (_, _, r) in
                                 self.trainset.all_ratings()])

        for u, i, r in trainset.all_rating():
	    # A[u, i] = r
            if r > the_mean:
	        A[u, i] = 1

        W = sparse.lil_matrix((item_num, item_num))

	for j in range(item_num):

	    aj = sparse.csc_matrix(A.getcol(j))
	    Aj = A.copy()
	    Aj[:, j] = 0
	    Aj = Aj.tocsc()

	    __, coefs, __ = linear_model.enet_path(Aj, aj, l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, alphas=alphas, positive=positive, max_iter=max_iter))

	    W[:, j]=coefs.reshape(item_num, 1)

	self.estimator=A.tocsc() * W.tocsc()


    def estimate(self, u, i):

	print("the output is mean value")
        return self.the_mean



    def predict_topn(self, uid, iid, t uidopn, verbose = False):
        """ Not implement self.estimate

        And generate recommendation list of a user here
        """
        # Convert raw ids to inner ids
        try:
            iuid=self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid='UKN__' + str(uid)
        try:
            iiid=self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid='UKN__' + str(iid)

        details={}
        try:

            user_rating=self.estimator[iuid, :].toarray()

            index_rating=[(i, user_rating[i])
                            for i in range(len(user_rating))]

            index_rating.sort(key=lambda x: -x[1])

            est_list=[x[0] for x in index_rating[:topn]]

            details['was_impossible']=False

        except PredictionImpossible as e:
            est_list=[]
            details['was_impossible']=True
            details['reason']=str(e)

        pred=Prediction_topn(uid, iiid, est_list, details)

        if verbose:
            print(pred)

        return pred

    def test_topn(self, testset, topn, verbose=False):
        predictions_topn=[
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
    data= Dataset.load_builtin('ml-100k')
    algo= MyOwnAlgorithm(
        l1_ratio = 1,
        eps = 1e-3,
        n_alphas = 100,
        # alphas=[10],
        max_iter = 10000)

    # evaluate(algo, data)
    evaluate_topn(algo, data, 10)
"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

import surprise as env
import MyEvaluate_v1 as topn
from scipy import sparse
from sklearn import linear_model


class SLIM(env.AlgoBase):

    def __init__(
            self,
            l1_ratio=0.5,
            n_alphas=10,
            alphas=None,
            threshold=3,
            eps=1e-3,
            max_iter=1000,
            positive=True):

        # Always call base method before doing anything.
        env.AlgoBase.__init__(self)
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.eps = eps
        self.max_iter = max_iter
        self.positive = positive
        self.threshold = threshold
        self.estimator = None

    def train(self, trainset):

        # Here again: call base method before doing anything.
        env.AlgoBase.train(self, trainset)

        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        A = sparse.lil_matrix((user_num, item_num))

        for u, i, r in trainset.all_ratings():
            # A[u, i] = r
            if r > self.threshold:
                A[u, i] = 1

        W = sparse.lil_matrix((item_num, item_num))

        for j in range(item_num):

            aj = A.getcol(j).toarray().ravel()
            Aj = A.copy()
            Aj[:, j] = 0
            Aj = Aj.tocsc()
            __, coefs, __ = linear_model.enet_path(Aj, aj, l1_ratio=self.l1_ratio, eps=self.eps,
                                                   n_alphas=self.n_alphas, alphas=self.alphas,
                                                   positive=self.positive, max_iter=self.max_iter)

            W[:, j] = coefs[:, -1].reshape(item_num, 1)

        self.estimator = A.tocsc() * W.tocsc()

    def estimate(self, u, i):

        return self.estimator[u, i]


if __name__ == '__main__':

    # builtin dataset
    # data = env.Dataset.load_builtin('ml-100k')

    # ===============================  load data  ===================================
    # ml-latest-small
    # file_path = 'input/ml-latest-small/ratings.csv'
    # reader = env.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # ------------------------------------------------------------------------------
    # ml-100k
    file_path = 'input/ml-100k/u.data'
    reader = env.Reader(line_format='user item rating timestamp', sep='\t', skip_lines=1)
    # ------------------------------------------------------------------------------
    # ml-20m
    # file_path = 'input/ml-20m/ratings.csv'
    # reader = env.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # ==============================================================================

    data = env.Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=3)

    algo = SLIM(
        l1_ratio=1,
        eps=1e-3,
        n_alphas=10,
        threshold=3,
        # alphas=[10],
        max_iter=5000,
        positive=True)

    topn.evaluate_topn(algo, data, top_n=100, threshold=4.5)

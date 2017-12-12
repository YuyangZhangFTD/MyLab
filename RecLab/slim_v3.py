import numpy as np
import surprise as env
from scipy import sparse
from sklearn.linear_model import SGDRegressor

import MyDataset as myDataset
import MyEvaluate_v2 as myEvaluate


class SLIM3(env.AlgoBase):
    """
        Bayesian Personalized Ranking
    """

    def __init__(self, l1_reg, l2_reg, max_iter, tol):
        env.AlgoBase.__init__(self)

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.tol = tol
        self.est = None

    def train(self, trainset):

        env.AlgoBase.train(self, trainset)

        alpha = self.l1_reg + self.l2_reg
        l1_ratio = self.l1_reg / alpha

        model = SGDRegressor(
            penalty='elasticnet',
            fit_intercept=False,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol
        )

        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        A = sparse.lil_matrix((user_num, item_num))

        for u, i, r in self.trainset.all_ratings():
            A[u, i] = r

        W = sparse.lil_matrix((item_num, item_num))

        for j in range(item_num):
            aj = A[:, j].copy()
            A[:, j] = 0

            model.fit(A, aj.toarray().ravel())

            w = model.coef_
            A[:, j] = aj

            w[w < 0] = 0
            for el in w.nonzero()[0]:
                W[(el, j)] = w[el]

        # print('~'*50)
        # print(W.nnz)
        # print(A.nnz)
        self.est = np.dot(A, W)

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            # print('unknown input: u-->' + str(u) + '  i-->' + str(i))
            raise env.PredictionImpossible('User and/or item is unkown.')

        return self.est[u, i]


if __name__ == '__main__':
    # builtin dataset
    # data = env.Dataset.load_builtin('ml-100k')

    # ===============================  load data  ===================================
    # ml-latest-small
    # file_path = 'input/ml-latest-small/ratings.csv'
    # reader = env.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # ------------------------------------------------------------------------------
    # ml-100k
    # file_path = 'input/ml-100k/u.data'
    # reader = env.Reader(line_format='user item rating timestamp', sep='\t', skip_lines=1)
    # ------------------------------------------------------------------------------
    # ml-20m
    # file_path = 'input/ml-20m/ratings.csv'
    # reader = env.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # ==============================================================================

    # data = env.Dataset.load_from_file(file_path, reader=reader)
    # data.split(n_folds=5)

    file_path = 'input/ml-100k/u.data'
    reader = myDataset.Reader(line_format='user item rating timestamp', sep='\t', skip_lines=1, implicit=True,
                              threshold=4.5)
    data = myDataset.Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=5)

    # define algorithm
    algo = SLIM3(l1_reg=0.001, l2_reg=0.01, max_iter=200, tol=1e-3)

    # evaluate
    # env.evaluate(algo, data, measures=['rmse', 'mae', 'fcp'])
    myEvaluate.evaluate(algo, data, measures=['fcp', 'hr', 'arhr'], topN=10, leave_out_num=5, verbose=1)

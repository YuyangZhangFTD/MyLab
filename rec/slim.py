from scipy import sparse
import numpy as np

import surprise as env


class SLIM(env.AlgoBase):

    def __init__(self, learning_rate=0.01, max_iter=100, alpha=0.5, reg=0.1):

        env.AlgoBase.__init__(self)
        self.eta = learning_rate
        self.maxiter = max_iter
        self.alpha = alpha
        self.reg = reg


    def train(self, trainset):

        env.AlgoBase.train(self, trainset)
        user_num = self.trainset.n_users
        item_num = self.trainset.n_items

        lil_rating = sparse.lil_matrix((user_num, item_num))

        for u, i, r in self.trainset.all_ratings():
            lil_rating[u, i] = r

        A = sparse.csc_matrix(lil_rating)
        W = sparse.lil_matrix((item_num, item_num))

        del lil_rating

        for j in range(item_num):

            wj = sparse.csc_matrix((item_num, 1))
            aj = A.getcol(j).toarray().ravel()
            Aj = A.copy()
            Aj[:, j] = 0
            for iter_i in range(self.maxiter):

                pass



    def estimate(self, u, i):
        try:
            estimator = 3
        except BaseException:
            print('unknown input: u-->' + str(u) + '  i-->' + str(i))
            estimator = 3
        return estimator


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
    data.split(n_folds=5)

    # define algorithm
    algo = SLIM()

    # evaluate
    env.evaluate(algo, data)

import numpy as np
import surprise as env
from scipy import sparse
from scipy.sparse.linalg import spsolve

import MyDataset as myDataset
import MyEvaluate_v2 as myEvaluate


class ALS1(env.AlgoBase):
    """
        Bayesian Personalized Ranking
    """

    def __init__(self, rank=10, reg=0.01, iter_num=10):
        env.AlgoBase.__init__(self)

        self.k = rank
        self.iter_num = iter_num
        self.reg = reg
        self.est = None

    def train(self, trainset):
        env.AlgoBase.train(self, trainset)

        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        P = np.random.random((user_num, self.k))  / 10
        Q = np.random.random((item_num, self.k))  / 10

        lil_rating = sparse.lil_matrix((user_num, item_num))

        for u, i, r in self.trainset.all_ratings():
            lil_rating[u, i] = r

        dok = lil_rating.todok()

        P = sparse.csc_matrix(P)
        Q = sparse.csc_matrix(Q)
        R = lil_rating.tocsc()
        regI = self.reg * sparse.eye(self.k).tocsc()

        for iter in range(self.iter_num):
            P = spsolve(Q.T * Q + regI, (R * Q).T).T
            Q = spsolve(P.T * P + regI, P.T * R).T

            # err = np.sum(np.power(R - np.dot(P, Q.T), 2))
            est = P * Q.T
            err = 0
            for (u, i), r in dok.items():
                err += (est[u, i] - r) ** 2

            print("=" * 50)
            print("iter:   " + str(iter))
            print("error:  " + str(err))
            print("loss:   " + str(err + np.sum(np.dot(P.T, P)) + np.sum(np.dot(Q.T, Q))))

        self.est = np.dot(P, Q.T)
        print(self.est)

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
    file_path = 'input/ml-100k/u.data'
    reader = env.Reader(line_format='user item rating timestamp', sep='\t', skip_lines=1)
    # ------------------------------------------------------------------------------
    # ml-20m
    # file_path = 'input/ml-20m/ratings.csv'
    # reader = env.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # ==============================================================================

    data = env.Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=5)

    # file_path = 'input/ml-100k/u.data'
    # reader = myDataset.Reader(line_format='user item rating timestamp', sep='\t', skip_lines=1, implicit=True,
    #                           threshold=4.5)
    # data = myDataset.Dataset.load_from_file(file_path, reader=reader)
    # data.split(n_folds=5)

    # define algorithm
    algo = ALS1(rank=30, iter_num=100, reg=0.5)

    # evaluate
    env.evaluate(algo, data, measures=['rmse', 'mae', 'fcp'])
    # myEvaluate.evaluate(algo, data, measures=['fcp', 'hr', 'arhr'], topN=10, leave_out_num=5, verbose=1)

import numpy as np
import surprise as env
from scipy import sparse

import MySurpriseEnv as myenv
from algo_base_rank import RankAlgoBase


def _sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


class BPR(RankAlgoBase):
    def __init__(
            self,
            learning_rate=0.00001,
            factor_num=20,
            max_iter=100,
            alpha=0.01,
            batch=100):

        RankAlgoBase.__init__(self)
        self.eta = learning_rate
        self.k = factor_num
        self.maxiter = max_iter
        self.reg = alpha
        self.batch = batch
        self.P = None  # P is user vector
        self.Q = None  # Q is item vector)

    def train(self, trainset):

        RankAlgoBase.train(self, trainset)

        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        lil_rating = sparse.lil_matrix((user_num, item_num))

        for u, i, r in self.trainset.all_ratings():
            lil_rating[u, i] = r

        # rating = sparse_rating.toarray()

        self.P = np.zeros((user_num, self.k)) + 0.1
        self.Q = np.zeros((item_num, self.k)) + 0.1

        # to dok_matrix for convenience
        dok_rating = sparse.dok_matrix(lil_rating)

        for iter_i in range(self.maxiter):
            loss = 0
            u_list = np.random.choice(user_num, self.batch)
            for u in u_list:
                item_list = dok_rating.getrow(u)
                num = item_list.nnz
                # e.g. [((0,0), 3.0), ((0,1), 2.0), ...]
                item_list = list(item_list.items())
                # index of item of sparse matrix
                i = np.random.randint(num)
                j = np.random.randint(num)
                if i == j or item_list[i][-1] == item_list[j][-1]:
                    continue
                # convert index of rating value sparse matrix
                # to index of rating matrix
                if item_list[i][-1] < item_list[j][-1]:
                    i, j = item_list[j][0][-1], item_list[i][0][-1]
                else:
                    i, j = item_list[i][0][-1], item_list[j][0][-1]

                s = _sigmoid(np.dot(self.P[u, :], self.Q[i, :]) -
                             np.dot(self.P[u, :], self.Q[j, :]))

                self.P[u, :] += self.eta * \
                                (1 - s) * (self.Q[i, :] - self.Q[j, :])
                self.Q[i, :] += self.eta * (1 - s) * self.P[u, :]
                self.Q[j, :] -= self.eta * (1 - s) * self.P[u, :]

                self.P[u, :] -= self.eta * self.reg * self.P[u, :]
                self.Q[i, :] -= self.eta * self.reg * self.Q[i, :]
                self.Q[j, :] -= self.eta * self.reg * self.Q[j, :]
                loss += -np.log(s)
            loss += self.reg * np.sum(self.P ** 2) + \
                    self.reg * np.sum(self.Q ** 2)
            print("iteration at " + str(iter_i + 1) + "  loss: " + str(loss))

        # estimator
        self.estimator = np.dot(self.P, self.Q.T)


if __name__ == '__main__':
    # builtin dataset
    # data = env.Dataset.load_builtin('ml-100k')

    # ===============================  load data  ============================
    # ml-latest-small
    # file_path = 'input/ml-latest-small/ratings.csv'
    # reader = env.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # ------------------------------------------------------------------------------
    # ml-100k
    file_path = 'input/ml-100k/u.data'
    reader = env.Reader(
        line_format='user item rating timestamp',
        sep='\t',
        skip_lines=1)
    # ------------------------------------------------------------------------------
    # ml-20m
    # file_path = 'input/ml-20m/ratings.csv'
    # reader = env.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # ==============================================================================

    data = env.Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=5)

    # define algorithm
    algo = BPR(learning_rate=0.01,
               factor_num=30,
               max_iter=200,
               alpha=0.01,
               batch=500)

    # evaluate
    myenv.evaluate_topn(algo, data, 10)

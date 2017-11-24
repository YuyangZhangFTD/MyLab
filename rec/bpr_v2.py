import surprise as env
import MyEvaluate as topn
import numpy as np
from scipy import sparse


def _sigmoid(x):

    return 1 / (1 + np.exp(-1 * x))


class BPR4TOPN(env.AlgoBase):
    def __init__(
            self,
            learning_rate=0.00001,
            factor_num=20,
            max_iter=100,
            batch=128,
            alpha=0.01,
            eps=1e-4,
            random=True):
        env.AlgoBase.__init__(self)
        self.eta = learning_rate
        self.k = factor_num
        self.maxiter = max_iter
        self.reg = alpha
        self.eps = eps
        self.batch = batch
        self.random = random
        self.mean = 0
        self.P = None  # P is user vector
        self.Q = None  # Q is item vector

    def train(self, trainset):

        env.AlgoBase.train(self, trainset)

        self.mean = self.trainset.global_mean

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
            # for u in range(user_num):
            for __ in range(self.batch):
                u = np.random.randint(user_num)

                if self.random:
                    u = np.random.randint(user_num)

                item_list = dok_rating.getrow(u)

                num = item_list.nnz

                # e.g. [((0,0), 3.0), ((0,1), 2.0), ...]
                item_list = list(item_list.items())

                for __ in range(num):

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

    def estimate(self, u, i):
        try:
            estimator = np.dot(self.P[u, :], self.Q[i, :])
        except BaseException:
            print('unknown input: u-->' + str(u) + '  i-->' + str(i))
            estimator = self.mean
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
    algo = BPR4TOPN(
        learning_rate=0.01,
        factor_num=50,
        max_iter=100,
        batch=512,
        alpha=0.01,
        eps=1e-2,
        random=False)

    # evaluate
    # topn.evaluate_topn(algo, data, top_n=100, threshold=4.5)
    env.evaluate(algo, data, measures=['fcp'])
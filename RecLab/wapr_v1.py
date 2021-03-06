import numpy as np
import surprise as env
from scipy import sparse


def _sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


class WAPR(env.AlgoBase):

    """
        WAPR, BPR without updating negative sample weight
    """

    def __init__(
            self,
            learning_rate=0.00001,
            factor_num=20,
            epoch_num=5,
            batch_num=512,
            alpha=0.01,
            eps=1e-4,
            random=True):

        env.AlgoBase.__init__(self)

        self.eta = learning_rate
        self.k = factor_num
        self.epoch = epoch_num
        self.batch = batch_num
        self.reg = alpha
        self.eps = eps
        self.random = random
        self.mean = 0
        self.est = None

    def train(self, trainset):

        env.AlgoBase.train(self, trainset)

        self.mean = self.trainset.global_mean

        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        lil_rating = sparse.lil_matrix((user_num, item_num))

        for u, i, r in self.trainset.all_ratings():
            lil_rating[u, i] = r

        # rating = sparse_rating.toarray()

        P = np.zeros((user_num, self.k)) + 0.1
        Q = np.zeros((item_num, self.k)) + 0.1

        # to dok_matrix for convenience
        dok_rating = sparse.dok_matrix(lil_rating)

        num = dok_rating.nnz
        rating_list = list(dok_rating.items())

        for epoch_i in range(self.epoch):
            print("-" * 20 + "epoch:  " + str(epoch_i + 1) + "-" * 20)
            batch_i = 1
            loss = 0

            for sample_i in range(num):

                if sample_i % self.batch == 0:
                    print("batch:  " + str(batch_i))
                    batch_i += 1

                # get train pair randomly
                # pair = np.random.randint(num)

                (u, i), _ = rating_list[sample_i]
                for __ in range(self.implicitNum):
                    j = np.random.randint(item_num)

                    s = _sigmoid(np.dot(P[u, :], Q[i, :]) - np.dot(P[u, :], Q[j, :]))

                    # the negative sample won't be updated

                    P[u, :] += self.eta * (1 - s) * (Q[i, :] - Q[j, :])
                    Q[i, :] += self.eta * (1 - s) * P[u, :]
                    # Q[j, :] -= self.eta * (1 - s) * P[u, :]

                    P[u, :] -= self.eta * self.reg * P[u, :]
                    Q[i, :] -= self.eta * self.reg * Q[i, :]
                    # Q[j, :] -= self.eta * self.reg * Q[j, :]

                loss += np.log(s)

            loss -= self.reg * (np.sum(P ** 2) + np.sum(Q ** 2))
            print("Epoch iteration at " + str(epoch_i) + "  loss: " + str(loss))

        self.est = np.dot(Q, P.T).T

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            print('unknown input: u-->' + str(u) + '  i-->' + str(i))
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

    # define algorithm
    algo = WAPR(
        learning_rate=0.01,
        factor_num=20,
        epoch_num=1,
        batch_num=512,
        alpha=0.01,
        eps=1e-2,
        random=False)

    # evaluate
    # topn.evaluate_topn(algo, data, top_n=100, threshold=4.5)
    env.evaluate(algo, data, measures=['fcp'])

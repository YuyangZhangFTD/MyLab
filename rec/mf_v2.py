import numpy as np
import surprise as env
from scipy import sparse


class MF2(env.AlgoBase):
    """
        train factor in a new way
        for f in range(n_factors):
            for _ in range(n_iter):
                for u, i, r in all_ratings:
                    err = r_ui - <p[u, :f+1], q[i, :f+1]>
                    update p[u, f]
                    update q[i, f]
    """

    def __init__(
            self,
            factor_num=10,
            max_iter=500,
            learning_rate=0.001,
            reg=0.1,
            batch_size=100,
            sgd=False,
            bias=True):

        env.AlgoBase.__init__(self)
        self.k = factor_num
        self.maxiter = max_iter
        self.eta = learning_rate
        self.ifbias = bias
        self.reg = reg
        self.batch = batch_size
        self.withsgd = sgd
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.mu = None

    def train(self, trainset):

        env.AlgoBase.train(self, trainset)
        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        self.mu = self.trainset.global_mean
        self.bu = np.zeros([user_num, 1])
        self.bi = np.zeros([item_num, 1])
        self.P = np.zeros((user_num, self.k)) + 0.1
        self.Q = np.zeros((item_num, self.k)) + 0.1

        lil_rating = sparse.lil_matrix((user_num, item_num))

        for u, i, r in self.trainset.all_ratings():
            lil_rating[u, i] = r

        dok_rating = sparse.dok_matrix(lil_rating)
        rating_num = dok_rating.nnz
        uir_list = list(dok_rating.items())

        for f in range(self.k):
            print("-" * 12 + str(f) + "-" * 12)
            for iter_i in range(self.maxiter):
                square_loss = 0

                if self.withsgd:
                    # Stochastic Gradient Descent
                    batch_index = np.random.choice(rating_num, self.batch)
                else:
                    # Gradient Decent for all data
                    batch_index = range(rating_num)

                for index in batch_index:
                    (u, i), r = uir_list[index]

                    hat = self.mu + self.bu[u] + self.bi[i] + \
                        np.dot(self.P[u, :f + 1], self.Q[i, :f + 1])
                    err = r - hat

                    if self.ifbias:
                        self.bu[u] += self.eta * (err - self.reg * self.bu[u])
                        self.bi[i] += self.eta * (err - self.reg * self.bi[i])

                    self.P[u, :f + 1] += self.eta * \
                                         (err * self.Q[i, :f + 1] - self.reg * self.P[u, :f + 1])
                    self.Q[i, :f + 1] += self.eta * \
                                         (err * self.P[u, :f + 1] - self.reg * self.Q[i, :f + 1])
                    square_loss += (r - hat)**2
                loss = 0.5 * square_loss + self.reg * \
                    (np.sum(self.bu**2) + np.sum(self.bi**2) + np.sum(self.P**2) + np.sum(self.Q**2))
                print("iteration at " + str(iter_i+1) + "  loss: " + str(loss))

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise env.PredictionImpossible('User and/or item is unkown.')
            print('unknown input: u-->' + str(u) + '  i-->' + str(i))

        estimator = np.dot(self.P[u, :], self.Q[i, :])
        if self.ifbias:
            estimator += self.mu + self.bu[u] + self.bi[i]


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
    algo = MF2(factor_num=100,
               max_iter=1000,
               learning_rate=0.0001,
               reg=0.1,
               batch_size=100,
               sgd=False,
               bias=True)

    # evaluate
    env.evaluate(algo, data)

import numpy as np
import surprise as env
from scipy import sparse


class MF(env.AlgoBase):
    def __init__(
            self,
            factor_num=10,
            max_iter=200,
            learning_rate=0.001,
            reg=0.1,
            batch_size=100,
            sgd=True,
            bias=True):

        env.AlgoBase.__init__(self)
        self.k = factor_num
        self.maxiter = max_iter
        self.eta = learning_rate
        self.ifbias = bias
        self.withsgd = sgd
        self.reg = reg
        self.batch = batch_size
        self.est = None
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.mu = None

    def train(self, trainset):

        env.AlgoBase.train(self, trainset)
        self.mu = self.trainset.global_mean
        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        bu = np.zeros([user_num, 1])
        bi = np.zeros([item_num, 1])
        P = np.zeros((user_num, self.k)) + 0.1
        Q = np.zeros((item_num, self.k)) + 0.1

        lil_rating = sparse.lil_matrix((user_num, item_num))

        for u, i, r in self.trainset.all_ratings():
            lil_rating[u, i] = r

        dok_rating = sparse.dok_matrix(lil_rating)

        rating_num = dok_rating.nnz
        uir_list = list(dok_rating.items())

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

                hat = self.mu + bu[u] + bi[i] + np.dot(P[u, :], Q[i, :])
                err = r - hat

                if self.ifbias:
                    bu[u] += self.eta * (err - self.reg * bu[u])
                    bi[i] += self.eta * (err - self.reg * bi[i])

                P[u, :] += self.eta * (err * Q[i, :] - self.reg * P[u, :])
                Q[i, :] += self.eta * (err * P[u, :] - self.reg * Q[i, :])
                square_loss += (r - hat) ** 2
            loss = 0.5 * square_loss + self.reg * (np.sum(bu ** 2) + np.sum(bi ** 2) + np.sum(P ** 2) + np.sum(Q ** 2))
            print("iteration at " + str(iter_i + 1) + "  loss: " + str(loss))

        estimator = np.dot(Q, P.T)
        self.est = estimator
        self.bu = bu
        self.bi = bi

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

    data = env.Dataset.load_from_file(file_path, reader=reader)  # for rating
    data.split(n_folds=5)

    # define algorithm
    algo = MF(factor_num=100,
              max_iter=200,
              learning_rate=0.001,
              reg=0.1,
              batch_size=100,
              sgd=False,
              bias=True)

    # evaluate
    env.evaluate(algo, data)

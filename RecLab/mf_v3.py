import numpy as np
import surprise as env
from scipy import sparse


class MF3(env.AlgoBase):
    """
        batch gradient boosting matrix factorization
    """
    def __init__(
            self,
            factor_num=10,
            max_iter=500,
            learning_rate=0.001,
            reg=0.1,
            batch_size=100,
            batch_factor=1,
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
        self.step = batch_factor
        self.est = None
        self.mu = None
        self.bu = None
        self.bi = None

    def train(self, trainset):

        env.AlgoBase.train(self, trainset)
        user_num = self.trainset.n_users
        item_num = self.trainset.n_items
        mu = self.trainset.global_mean
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

        for f in range(0, self.k, self.step):
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

                    hat = mu + bu[u] + bi[i] + \
                          np.dot(P[u, :f + 1], Q[i, :f + 1])
                    err = r - hat

                    if self.ifbias:
                        bu[u] += self.eta * (err - self.reg * bu[u])
                        bi[i] += self.eta * (err - self.reg * bi[i])

                    P[u, f - self.step:f + 1] += self.eta * (
                        err * Q[i, f - self.step:f + 1] - self.reg * P[u, f - self.step:f + 1])
                    Q[i, f - self.step:f + 1] += self.eta * (
                        err * P[u, f - self.step:f + 1] - self.reg * Q[i, f - self.step:f + 1])
                    square_loss += (r - hat) ** 2
                loss = 0.5 * square_loss + self.reg * (
                    np.sum(bu ** 2) + np.sum(bi ** 2) + np.sum(P ** 2) + np.sum(Q ** 2))
                print("iteration at " + str(iter_i + 1) + "  loss: " + str(loss))

        estimator = np.dot(P, Q.T)
        self.est = estimator
        self.mu = mu
        self.bu = bu
        self.bi = bi

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            print('unknown input: u-->' + str(u) + '  i-->' + str(i))
            raise env.PredictionImpossible('User and/or item is unkown.')

        bias = self.mu + self.bu[u] + self.bi[i] if self.ifbias else 0
        return self.est[u, i] + bias


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
    algo = MF3(factor_num=20,
               max_iter=100,
               learning_rate=0.001,
               reg=0.1,
               batch_size=100,
               batch_factor=2,
               sgd=False,
               bias=True)

    # evaluate
    env.evaluate(algo, data, measures=['rmse', 'mae', 'fcp'])

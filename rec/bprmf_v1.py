import numpy as np
import surprise as env
from scipy import sparse


def _sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


class BPRMF(env.AlgoBase):
    """
        alternative loss between mf-rating loss and bpr-ranking loss
    """

    def __init__(
            self,
            factor_num=10,
            max_iter_mf=200,
            max_iter_bpr=100,
            epoch_n=10,
            learning_rate=0.001,
            reg=0.1,
            batch_size=100,
            withsgd=True,
            bias=True):

        env.AlgoBase.__init__(self)
        self.k = factor_num
        self.mfmaxiter = max_iter_mf
        self.bprmaxiter = max_iter_bpr
        self.eta = learning_rate
        self.ifbias = bias
        self.reg = reg
        self.withsgd = withsgd
        self.maxepoch = epoch_n
        self.batch = batch_size
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

        for epoch_i in range(self.maxepoch):
            print("Epoch iteration at: " + str(epoch_i))

            # mf
            print("=" * 20 + "Train rating loss" + "=" * 20)
            for iter_i in range(self.mfmaxiter):
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
                          np.dot(self.P[u, :], self.Q[i, :])
                    err = r - hat

                    if self.ifbias:
                        self.bu[u] += self.eta * (err - self.reg * self.bu[u])
                        self.bi[i] += self.eta * (err - self.reg * self.bi[i])

                    self.P[u, :] += self.eta * \
                                    (err * self.Q[i, :] - self.reg * self.P[u, :])
                    self.Q[i, :] += self.eta * \
                                    (err * self.P[u, :] - self.reg * self.Q[i, :])
                    square_loss += (r - hat) ** 2
                loss = 0.5 * square_loss + self.reg * \
                                           (np.sum(self.bu ** 2) + np.sum(self.bi ** 2) + np.sum(self.P ** 2) + np.sum(
                                               self.Q ** 2))
                print("iteration at " + str(iter_i + 1) +
                      "  batch loss: " + str(loss))

            # bpr
            print("=" * 20 + "Train ranking loss" + "=" * 20)
            for iter_i in range(self.bprmaxiter):
                square_loss = 0

                batch_index = np.random.choice(rating_num, self.batch)
                for index in batch_index:
                    (u, i), r = uir_list[index]

                    # e.g. [((0,0), 3.0), ((0,1), 2.0), ...]
                    item_list = dok_rating.getrow(u)
                    num = item_list.nnz
                    item_list = list(item_list.items())

                    # index of item of sparse matrix
                    i = np.random.randint(num)
                    j = np.random.randint(num)

                    if i == j or item_list[i][-1] == item_list[j][-1]:
                        continue

                    # convert index of rating value sparse matrix
                    # to index of rating matrix
                    if item_list[i][-1] < item_list[j][-1]:
                        k = j
                        i, j = item_list[j][0][-1], item_list[i][0][-1]
                    else:
                        k = i
                        i, j = item_list[i][0][-1], item_list[j][0][-1]

                    hat_ui = np.dot(self.P[u, :], self.Q[i, :]) + self.bi[i]
                    hat_uj = np.dot(self.P[u, :], self.Q[j, :]) + self.bi[j]

                    s = _sigmoid(hat_ui - hat_uj)

                    # TODO update bi

                    # update P and Q
                    self.P[u, :] += self.eta * \
                                    (1 - s) * (self.Q[i, :] - self.Q[j, :])
                    self.Q[i, :] += self.eta * (1 - s) * self.P[u, :]
                    self.Q[j, :] -= self.eta * (1 - s) * self.P[u, :]

                    self.P[u, :] -= self.eta * self.reg * self.P[u, :]
                    self.Q[i, :] -= self.eta * self.reg * self.Q[i, :]
                    self.Q[j, :] -= self.eta * self.reg * self.Q[j, :]

                    hat_ui = self.mu + self.bu[u] + self.bi[i] + \
                             np.dot(self.P[u, :], self.Q[i, :])
                    square_loss += (item_list[k][-1] - hat_ui) ** 2
                loss = 0.5 * square_loss + self.reg * \
                                           (np.sum(self.bu ** 2) + np.sum(self.bi ** 2) + np.sum(self.P ** 2) + np.sum(
                                               self.Q ** 2))
                print("user iteration at " + str(iter_i + 1) +
                      "  batch loss: " + str(loss))

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            print('unknown input: u-->' + str(u) + '  i-->' + str(i))
            raise env.PredictionImpossible('User and/or item is unkown.')

        estimator = np.dot(self.P[u, :], self.Q[i, :])
        if self.ifbias:
            estimator += self.mu + self.bu[u] + self.bi[i]


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
    algo = BPRMF(factor_num=10,
                 max_iter_mf=200,
                 max_iter_bpr=10,
                 learning_rate=0.001,
                 reg=0.1,
                 batch_size=1000,
                 withsgd=True,
                 bias=True)

    # evaluate
    env.evaluate(algo, data)

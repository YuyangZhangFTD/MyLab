import surprise as env
import numpy as np
import MySurpriseEnv as myenv
from scipy import sparse
from algo_base_rank import RankAlgoBase


class MfRanking(RankAlgoBase):
    def __init__(self,
                 factor_num=10,
                 max_iter=200,
                 learning_rate=0.001,
                 reg=0.1,
                 batch_size=100,
                 sgd=True,
                 bias=True):

        self.k = factor_num
        self.maxiter = max_iter
        self.eta = learning_rate
        self.ifbias = bias
        self.withsgd = sgd
        self.reg = reg
        self.batch = batch_size
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.mu = None
        self.estimator = None

    def train(self, trainset):

        self.trainset = trainset
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
            print("iteration at " + str(iter_i + 1) + "  loss: " + str(loss))

        if user_num < 1000000 and item_num < 1000000:
            self.estimator = np.dot(self.P, self.Q.T)
            if self.ifbias:
                self.estimator += self.mu + self.bu * \
                    np.ones([user_num, item_num]) + (self.bi * np.ones([item_num, user_num])).T

    def predict(self, uid, iid, topn, verbose=False):
        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            if self.estimator is not None:
                user_rating = self.estimator[iuid, :]
            else:
                user_rating = np.dot(self.P[iuid, :], self.Q)
                if self.ifbias:
                    user_rating += self.mu + self.bu[iuid] + self.bi.T
            index_rating = [(i, user_rating[i])
                            for i in range(len(user_rating))]
            index_rating.sort(key=lambda x: -x[1])
            est_list = [x[0] for x in index_rating[:topn]]
            details['was_impossible'] = False
        except env.PredictionImpossible as e:
            est_list = []
            details['was_impossible'] = True
            details['reason'] = str(e)

        pred = myenv.Prediction_topn(iuid, iiid, est_list, details)
        if verbose:
            print(pred)
        return pred

    def test(self, testset, topn, rating_threshold=3, verbose=False):
        if self.estimator is None:
            print("estimator is not available")
            predictions_topn = None
        else:
            predictions_topn = [
                self.predict(uid, iid, topn, verbose=verbose)
                for (uid, iid, r_ui_trans) in testset if r_ui_trans > rating_threshold
            ]
        return predictions_topn


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
    algo = MfRanking(factor_num=10,
                     max_iter=200,
                     learning_rate=0.001,
                     reg=0.1,
                     batch_size=100,
                     sgd=True,
                     bias=True)

    # evaluate
    myenv.evaluate_topn(algo, data, 10)

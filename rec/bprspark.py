import numpy as np
import surprise as env

import MySurpriseEnv as myenv
from algo_base_rank import RankAlgoBase


class MyAlgo(RankAlgoBase):
    def __init__(self):
        RankAlgoBase.__init__(self)

    def train(self, trainset):
        RankAlgoBase.train(self, trainset)
        P = np.loadtxt("userMatrix.txt", delimiter=',')
        Q = np.loadtxt("itemMatrix.txt", delimiter=',')
        self.estimator = np.dot(P.T, Q)


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
    algo = MyAlgo()

    # evaluate
    myenv.evaluate_topn(algo, data, 10)

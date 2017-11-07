import surprise as env


class BPRMF(env.AlgoBase):

    def __init__(self):

        env.AlgoBase.__init__(self)

    def train(self, trainset):

        env.AlgoBase.train(self, trainset)

        user_num = self.trainset.n_users
        item_num = self.trainset.n_items



    def estimate(self, u, i):
        try:
            estimator = 3
        except BaseException:
            print('unknown input: u-->' + str(u) + '  i-->' + str(i))
            estimator = 3
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
    # file_path = 'input/ml-100k/u.data'
    # reader = env.Reader(line_format='user item rating timestamp', sep='\t', skip_lines=1)
    # ------------------------------------------------------------------------------
    # ml-20m
    file_path = 'input/ml-20m/ratings.csv'
    reader = env.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # ==============================================================================

    data = env.Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=5)

    # define algorithm
    algo = BPRMF()

    # evaluate
    env.evaluate(algo, data)

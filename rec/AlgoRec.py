import surprise as env


class MyAlgo(env.AlgoBase):

    def __init__(self):

        env.AlgoBase.__init__(self)

    def train(self, trainset):

        env.AlgoBase.train(self, trainset)


    def estimate(self, u, i):
        try:
            estimator = 3
        except:
            print('unknown input: u-->'+str(u)+'  i-->'+str(i))
            estimator = 3
        return estimator


if __name__ == '__main__':
    
    # builtin dataset
    # data = env.Dataset.load_builtin('ml-100k')

    # load data
    # Attention: there is no title line in input file
    file_path = 'input/ml-latest-small/ratings_surprise.csv'
    reader = env.Reader(line_format='user item rating timestamp', sep=',')
    data = env.Dataset.load_from_file(file_path, reader=reader) 
    data.split(n_folds=5)

    # define algorithm
    algo = MyAlgo()

    # evaluate 
    env.evaluate(algo, data)

import surprise as env

path = "input/ml-latest-small/ratings.csv"
reader = env.Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)

data = env.Dataset.load_from_file(path, reader=reader)
data.split(n_folds=3)

algo = env.SVD()

env.evaluate(algo, data)

from collections import defaultdict

import surprise as env

import MySurpriseEnv as myenv


def evaluate_topn(algo, data, top_n=10, measures=['hit', 'arhr'], threshold=3.5,
                  with_dump=False, dump_dir=None, verbose=1):
    performances = myenv.CaseInsensitiveDefaultDict(list)

    if verbose:
        print('Evaluating {0} of algorithm {1}.'.format(
            ', '.join((m.upper() for m in measures)),
            algo.__class__.__name__))
        print()

    for fold_i, (trainset, testset) in enumerate(data.folds()):

        if verbose:
            print('-' * 12)
            print('Fold ' + str(fold_i + 1))

        # train and test algorithm. Keep all rating predictions in a list
        algo.train(trainset)
        topnset = trainset.build_anti_testset()
        predictions = algo.test(topnset, verbose=(verbose == 2))

        top_n_dict = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n_dict[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n_dict.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n_dict[uid] = user_ratings[:top_n]

        hr = 0
        arhr = 0
        for u, i, r in testset:
            if r < threshold:
                continue
            iuid = trainset.to_inner_uid(u)
            iiid = trainset.to_inner_iid(i)

            userset = set()
            if iuid not in userset:
                userset.add(iuid)
                if iiid in top_n_dict[iuid]:
                    hr += 1
                    arhr += 1 / (top_n_dict[iuid].index(iiid) + 1)
        hr /= len(userset)
        arhr /= len(userset)

    if verbose:
        print('HR: {0:1.4f}'.format(hr))
        print('ARHR: {0:1.4f}'.format(arhr))

    return hr, arhr


def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


class DatasetTopn(env.dataset.DatasetAutoFolds):
    def __init__(self, ratings_file=None, reader=None, df=None, threshold=3.5):
        env.dataset.DatasetAutoFolds.__init__(self, ratings_file, reader, df)
        self.threshold = threshold

    def construct_testset(self, raw_testset):
        return [(ruid, riid, r_ui_trans)
                for (ruid, riid, r_ui_trans, _) in raw_testset if r_ui_trans > self.threshold]


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

    data = DatasetTopn(ratings_file=file_path, reader=reader, threshold=3.5)
    data.split(n_folds=5)

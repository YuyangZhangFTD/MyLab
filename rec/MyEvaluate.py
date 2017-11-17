import surprise as env
import MySurpriseEnv as myenv
import numpy as np
from collections import defaultdict


def evaluate_topn(
        algo,
        data,
        top_n=10,
        measures=['hr', 'arhr'],
    threshold=3.5,
    with_dump=False,
    dump_dir=None,
        verbose=1):
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
        # TODO: topnset should not contain those items which are not in trainset

        topnset = trainset.build_anti_testset()
        predictions = algo.test(topnset, verbose=(verbose == 2))

        top_n_dict = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            try:
                iuid = trainset.to_inner_uid(uid)
            except ValueError:
                print('UKN__' + str(uid))
                continue
            try:
                iiid = trainset.to_inner_iid(iid)
            except ValueError:
                print('UKN__' + str(iid))
                continue
            top_n_dict[iuid].append((iiid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n_dict.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n_dict[uid] = user_ratings[:top_n]

        hr = 0
        arhr = 0
        userset = set()

        for u, i, r in testset:
            if r < threshold:
                continue
            try:
                iuid = trainset.to_inner_uid(u)
            except ValueError:
                continue
            try:
                iiid = trainset.to_inner_iid(i)
            except ValueError:
                continue

            if iuid not in userset:
                userset.add(iuid)
                usertopn = [x[0] for x in top_n_dict[iuid]]
                if iiid in usertopn:
                    hr += 1
                    arhr += 1 / (usertopn.index(iiid) + 1)

        hr /= len(userset)
        arhr /= len(userset)

        if verbose:
            print('HR: {0:1.4f}'.format(hr))
            print('ARHR: {0:1.4f}'.format(arhr))

        performances["hr"].append(hr)
        performances["arhr"].append(arhr)


    if verbose:
        print('-' * 12)
        print('-' * 12)
        for measure in measures:
            print('Mean {0:4s}: {1:1.4f}'.format(
                  measure.upper(), np.mean(performances[measure])))
        print('-' * 12)
        print('-' * 12)

    return hr, arhr


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
    reader = env.Reader(line_format='user item rating timestamp', sep='\t', skip_lines=1)
    # ------------------------------------------------------------------------------
    # ml-20m
    # file_path = 'input/ml-20m/ratings.csv'
    # reader = env.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    # ==============================================================================

    data = env.Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=5)
    algo = env.SVDpp()

    # evaluate_topn(algo, data, top_n=100, threshold=3, verbose=1)
    env.evaluate(algo, data, measures=['rmse', 'mae', 'fcp'], verbose=1)

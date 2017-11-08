from collections import namedtuple, defaultdict

import numpy as np

from algo_base_rank import RankAlgoBase


class CaseInsensitiveDefaultDict(defaultdict):
    """From here:
        http://stackoverflow.com/questions/2082152/case-insensitive-dictionary.

        As pointed out in the comments, this only covers a few cases and we
        should override a lot of other methods, but oh well...

        Used for the returned dict, so that users can use perf['RMSE'] or
        perf['rmse'] indifferently.
    """

    def __setitem__(self, key, value):
        super(CaseInsensitiveDefaultDict, self).__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super(CaseInsensitiveDefaultDict, self).__getitem__(key.lower())


class Prediction_topn(
    namedtuple(
        'Prediction', [
            'uid', 'iid', 'est_list', 'details'])):

    __slots__ = ()  # for memory saving purpose

    def __str__(self):
        return "TODO"


def accuracy_topn(predictions_topn, verbose=True):
    """ Compute top-N accuracy
    .. math::
        \\text{HR} = \\frac{#hits}{#users}
        \\text{ARHR} = \\frac{1}{#users}\\sum^{#hits}_{i=1}\\frac{1}{p_i}

    """
    if not predictions_topn:
        raise ValueError('Prediction list is empty.')

    hr = 0
    arhr = 0
    test_user = set()

    for uid, iiid, est_list, __ in predictions_topn:

        # only one item each user in test set
        if uid not in test_user and iiid in est_list:
            test_user.add(uid)
            hr += 1
            arhr += 1 / (est_list.index(iiid) + 1)

    hr /= len(test_user)
    arhr /= len(test_user)

    if verbose:
        print('HR: {0:1.4f}'.format(hr))
        print('ARHR: {0:1.4f}'.format(arhr))

    return hr, arhr


def evaluate_topn(
        algo,
        data,
        topn,
        measures=['hr', 'arhr'],
        with_dump=False,
        dump_dir=None,
        verbose=1):
    if not isinstance(algo, RankAlgoBase):
        print("Your algorithm is not rank-model")
        return None

    performances = CaseInsensitiveDefaultDict(list)

    for fold_i, (trainset, testset) in enumerate(data.folds()):

        if verbose:
            print('-' * 12)
            print('Fold ' + str(fold_i + 1))

        algo.train(trainset)
        predictions_topn = algo.test_topn(
            testset, topn, verbose=(verbose == 2))

        hr, arhr = accuracy_topn(predictions_topn)

        performances['hr'].append(hr)
        performances['arhr'].append(arhr)

        # TODO
        if with_dump:
            pass

    if verbose:

        print('-' * 12)
        print('-' * 12)
        for measure in measures:
            print('Mean {0:4s}: {1:1.4f}'.format(
                  measure, np.mean(performances[measure])))
        print('-' * 12)

    return performances

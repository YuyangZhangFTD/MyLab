"""
Some modules will be used in surprise.

Target:

1. The evaluation of top-N recommendation.

2.

TODO:

1. Top-N accuracy, hr and arhr.

2. The evalutation method of top-N recommendation.

DONE:

1. Finish prediction for top-N recommendation.

2.

"""
from surprise import *


class Prediction_topn(
    namedtuple(
        'Prediction', [
            'uid', 'iid', 'est_list', 'details'])):

    """ A named tuple for storing the results of a prediction.

    Args:
        uid: The (raw) user id
        iid: The (raw) item id
        est_list (list): The list of iid which user may click
        details (dict): Stores additional details about the prediction that
        might be useful for later analysis.
    """

    __slots__ = ()  # for memory saving purpose


    def __str__(self):

        return "TODO"


def accuracy_topn(predictions_topn, verbose=Ture):
    """ Compute top-N accuracy

    .. math::
        \\text{HR} = \\frac{#hits}{#users}
        \\text{ARHR} = \\frac{1}{#users}\\sum^{#hits}_{i=1}\\frac{1}{p_i}

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Hit Rate and the Average Reciprocal Hit-Rank.

    Raises:
        ValueError: When ``predictions`` is empty.

    """
    if not predictions_topn:
        raise ValueError('Prediction list is empty.')

    # TODO: evaluate hr and arhr
    hr = 0
    arhr = 0
 
    test_user = set()

    for uid, iiid, est_list, __ in predictions_topn:

	test_user.add(uid)

	if iiid in est_list:
	    hr += 1
            arhr += 1/(est_list.index(iiid) + 1)
    

    hr /= len(test_user)
    arhr /= len(test_user)

    if verbose:
        print('HR: {0:1.4f}'.format(hr))
        print('ARHR: {0:1.4f}'.format(arhr))

    return hr, arhr


def evaluate_topn(algo, data, topn, measures=['hr', 'arhr'],with_dump=False, dump_dir=None, verbose=1):
    """ Evaluate the performance of the algorithm on the given data for top-N recommendation.

    Args:
        algo(:obj:`AlgoBase
            <surprise.prediction_algorithms.algo_base.AlgoBase>`):
            The algorithm to evaluate.
        data(:obj:`Dataset <surprise.dataset.Dataset>`): The dataset on which
            to evaluate the algorithm.
        topn(int): the number of items for recommendation.
        with_dump(bool): If True, the predictions and the algorithm will be
            dumped for later further analysis at each fold (see :ref:`FAQ
            <serialize_an_algorithm>`). The file names will be set as:
            ``'<date>-<algorithm name>-<fold number>'``.  Default is ``False``.
        dump_dir(str): The directory where to dump to files. Default is
            ``'~/.surprise_data/dumps/'``.
        verbose(int): Level of verbosity. If 0, nothing is printed. If 1
            (default), accuracy measures for each folds are printed, with a
            final summary. If 2, every prediction is printed.
    Returns:
        A dictionary containing measures as keys and lists as values. Each list
        contains one entry per fold.
    """

    performances = CaseInsensitiveDefaultDict(list)    

    for fold_i, (train, testset) in enumerate(data.folds()):

        if verbose:
            print('-' * 12)
            print('Fold ' + str(fold_i + 1))

        algo.train(trainset)
        predictions_topn = algo.test_topn(testset, topn, verbose=(verbose == 2))

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

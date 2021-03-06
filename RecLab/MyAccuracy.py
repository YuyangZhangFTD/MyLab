"""
The :mod:`surprise.accuracy` module provides with tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mae
    fcp
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
from six import iteritems
import random


def rmse(predictions, verbose=True, **kwargs):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est) ** 2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mae(predictions, verbose=True, **kwargs):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_


def fcp(predictions, verbose=True, **kwargs):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp_ = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp_))

    return fcp_


def hr(predictions, verbose=True, **kwargs):
    """Compute hit rate for implicit feedback

   #users is the total number of users, and #hits is the number of users
   whose item in the testing set is recommended (i.e., hit) in the size-N
   recommendation list.

    .. math::
        \\text{HR} = \\frac{#hits}{|#users|} \

    M. Deshpande and G. Karypis,
    Item-based top-n recommendation algorithms,
    ACM Transactions on Information Systems, vol. 22, pp.143–177, January 2004.

    Attention: in that paper, the author puts one non-zero item into test set
    randomly, and uses the remaining entries for training. While evaluating,
    the author checks whether the selected item is in the top-N list.
    For convenience here, I make a new test set consisted of one non-zero item
     and other zero items which are from the old test set.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    # if not predictions:
    #     raise ValueError('Prediction list is empty.')

    if not kwargs['topN'] and not kwargs['predictions_other'] and not kwargs['leave_out_num']:
        raise ValueError('Lack of parameters')
    else:
        topN = kwargs['topN']
        predictions_other = kwargs['predictions_other']
        leave_out_num = kwargs['leave_out_num']

    # ii_set        [(u, i), ...]
    # clicked_u     {u: [(i, est), ...]}
    # test_set      [(u, i, est), ...]
    # test_set_u    {u: [(i, est), ...]}

    ui_set = set()
    clicked_u = defaultdict(list)
    for u, i, r, est, _ in predictions:
        if int(r) == 1:
            clicked_u[u].append((i, est))
        ui_set.add((u, i))

    test_set = [(u, i, est) for (u, i, r, est, _) in predictions_other if (u, i) not in ui_set]

    test_set_u = defaultdict(list)
    for u, i, est in test_set:
        test_set_u[u].append((i, est))

    hit = 0
    record = 0
    for u, preds in iteritems(test_set_u):
        if len(clicked_u[u]) < leave_out_num:
            record += 1
            continue

        random.shuffle(clicked_u[u])
        chosen_list = clicked_u[u][:leave_out_num]
        preds.extend(chosen_list)
        preds.sort(key=lambda x: x[-1])
        topN_list = [tpl[0] for tpl in preds[:topN]]

        hit += sum([1 for click in chosen_list if click[0] in topN_list])

    hr_ = hit / (len(test_set_u.keys()) - record)

    if verbose:
        print('HR:  {0:1.4f}'.format(hr_))

    print('testset:  ' + str(len(predictions)) + '   not enough sample: ' + str(
        record))

    return hr_


def arhr(predictions, verbose=True, **kwargs):
    """Compute Average Reciprocal Hit-Rank for implicit feedback

    Attention: in that paper, the author puts one non-zero item into test set
    randomly, and uses the remaining entries for training. While evaluating,
    the author checks whether the selected item is in the top-N list.
    For convenience here, I make a new test set consisted of one non-zero item
     and other zero items which are from the old test set.

    .. math::
        \\text{ARHR} = \\frac{1}{|\\hat{#users}|}
        \\sum^{#hits}_{i=1}\\frac{1}{p_i}\

    M. Deshpande and G. Karypis,
    Item-based top-n recommendation algorithms,
    ACM Transactions on Information Systems, vol. 22, pp.143–177, January 2004.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    # if not predictions:
    #     raise ValueError('Prediction list is empty.')

    if not kwargs['topN'] and not kwargs['predictions_other'] and not kwargs['leave_out_num']:
        raise ValueError('Lack of parameters')
    else:
        topN = kwargs['topN']
        predictions_other = kwargs['predictions_other']
        leave_out_num = kwargs['leave_out_num']

    # ii_set        [(u, i), ...]
    # clicked_u     {u: [(i, est), ...]}
    # test_set      [(u, i, est), ...]
    # test_set_u    {u: [(i, est), ...]}

    ui_set = set()
    clicked_u = defaultdict(list)
    for u, i, r, est, _ in predictions:
        if int(r) == 1:
            clicked_u[u].append((i, est))
        ui_set.add((u, i))

    test_set = [(u, i, est) for (u, i, r, est, _) in predictions_other if (u, i) not in ui_set]

    test_set_u = defaultdict(list)
    for u, i, est in test_set:
        test_set_u[u].append((i, est))

    arhr_ = 0
    record = 0
    for u, preds in iteritems(test_set_u):
        if len(clicked_u[u]) < leave_out_num:
            record += 1
            continue

        random.shuffle(clicked_u[u])
        chosen_list = clicked_u[u][:leave_out_num]
        preds.extend(chosen_list)
        preds.sort(key=lambda x: x[-1])
        topN_list = [tpl[0] for tpl in preds[:topN]]

        try:
            arhr_ += sum([1 / (topN_list.index(click) + 1) for click in chosen_list if click[0] in topN_list])
        except ValueError:
            arhr_ += 0

    arhr_ = arhr_ / (len(test_set_u.keys()) - record)

    if verbose:
        print('ARHR:  {0:1.4f}'.format(arhr_))

    print('testset:  ' + str(len(predictions)) + '   not enough sample: ' + str(
        record))

    return arhr_


def _hr_arhr(predictions, verbose=True, **kwargs):
    # if not predictions:
    #     raise ValueError('Prediction list is empty.')

    if not kwargs['topN']:  # and not kwargs['predictions_topN']:
        raise ValueError('Lack of parameters')
    else:
        topN = kwargs['topN']
        # predictions = kwargs['predictions_topN']

    predictions_u = defaultdict(list)

    for u, i, r, est, _ in predictions:
        predictions_u[u].append((i, r, est))

    hit = 0
    arhr_ = 0
    record = 0

    for _, preds in iteritems(predictions_u):

        testset = []
        clicked_list = []
        for tple in preds:
            if tple[1] == 1:
                clicked_list.append(tple)
            else:
                testset.append(tple)

        if len(testset) < topN or len(clicked_list) < 1:
            record += 1
            continue

        clicked_one = clicked_list[np.random.randint(len(clicked_list))]
        testset.append(clicked_one)
        testset.sort(key=lambda x: x[-1])
        topN_list = testset[:topN:-1]

        hit += 1 if clicked_one in topN_list else 0
        try:
            arhr_ += 1 / (topN.index(clicked_one) + 1)
        except ValueError:
            arhr_ += 0

    hr_ = hit / (len(predictions_u.keys()) - record)
    arhr_ = arhr_ / (len(predictions_u.keys()) - record)

    if verbose:
        print('AR:   {0:1.4f}'.format(hr_))
        print('ARHR:  {0:1.4f}'.format(arhr_))

    print('testset:  ' + str(len(predictions)) + '   not enough sample: ' + str(
        record))

    return (hr_, arhr_)

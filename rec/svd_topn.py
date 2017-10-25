from collections import defaultdict
import numpy as np

from surprise import *

from MySurprise import *


class MySVD(SVD):

    def train(self, trainset):

        SVD.train(self, trainset)

        self.the_mean = np.mean(
            [r for (_, _, r) in self.trainset.all_ratings()])

    def predict_topn(self, uid, iid, topn, verbose=False):

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

            if self.biased:

                est = np.zeros(self.trainset.n_items)

                est += self.bu[iuid]

                est += self.bi

                est += np.dot(self.qi, self.pu[iuid])

            else:

                est = np.dot(self.qi, self.pu[iuid])

            user_rating = est.tolist()

            index_rating = [(i, user_rating[i])
                            for i in range(len(user_rating))]

            index_rating.sort(key=lambda x: -x[1])

            est_list = [x[0] for x in index_rating[:topn]]

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est_list = []
            details['was_impossible'] = True
            details['reason'] = str(e)

        pred = Prediction_topn(uid, iiid, est_list, details)

        if verbose:
            print(pred)

        return pred

    def test_topn(self, testset, topn, verbose=False):
        predictions_topn = [
            self.predict_topn(
                uid,
                iid,
                topn,
                verbose=verbose) for (
                uid,
                iid,
                r_ui_trans) in testset if r_ui_trans > 4]
        return predictions_topn


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


data = Dataset.load_builtin('ml-100k')
algo = MySVD()
evaluate_topn(algo, data, 20)





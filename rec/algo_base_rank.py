import surprise as env

import MySurpriseEnv as myenv


class RankAlgoBase(object):
    def __init__(self, **kwargs):

        self.bsl_options = kwargs.get('bsl_options', {})
        self.sim_options = kwargs.get('sim_options', {})
        self.trainset = None
        self.estimator = None
        self.bu = None
        self.bi = None
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based'] = True

    def train(self, trainset):

        self.trainset = trainset

        # self.estimator =

    # TODO
    def predict(self, uid, iid, topn, verbose=False):
        # Convert raw ids to inner ids
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
            user_rating = self.estimator[iuid, :]
            index_rating = [(i, user_rating[i])
                            for i in range(len(user_rating))]
            index_rating.sort(key=lambda x: -x[1])
            est_list = [x[0] for x in index_rating[:topn]]
            details['was_impossible'] = False
        except env.PredictionImpossible as e:
            est_list = []
            details['was_impossible'] = True
            details['reason'] = str(e)

        pred = myenv.Prediction_topn(iuid, iiid, est_list, details)
        if verbose:
            print(pred)
        return pred

    # TODO
    def test(self, testset, topn, rating_threshold=3, verbose=False):
        if self.estimator is None:
            print("estimator is not available")
            predictions_topn = None
        else:
            predictions_topn = [
                self.predict(uid, iid, topn, verbose=verbose)
                for (uid, iid, r_ui_trans) in testset if r_ui_trans > rating_threshold
            ]
        return predictions_topn

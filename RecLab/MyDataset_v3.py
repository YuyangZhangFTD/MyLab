"""
The :mod:`dataset <surprise.dataset>` module defines the :class:`Dataset` class
and other subclasses which are used for managing datasets.

Users may use both *built-in* and user-defined datasets (see the
:ref:`getting_started` page for examples). Right now, three built-in datasets
are available:

* The `movielens-100k <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `movielens-1m <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `Jester <http://eigentaste.berkeley.edu/dataset/>`_ dataset 2.

Built-in datasets can all be loaded (or downloaded if you haven't already)
using the :meth:`Dataset.load_builtin` method.
Summary:

.. autosummary::
    :nosignatures:

    Dataset.load_builtin
    Dataset.load_from_file
    Dataset.load_from_folds
    Dataset.folds
    DatasetAutoFolds.split
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import sys
import os
import itertools
import random
import warnings

from six.moves import input
from six.moves import range
from collections import defaultdict

from MyReader_v3 import Reader
from MyBuiltinDatasets_v3 import download_builtin_dataset
from MyBuiltinDatasets_v3 import BUILTIN_DATASETS
from MyTrainset_v3 import Trainset


class Dataset:
    """Base class for loading datasets.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the three
    available methods for loading datasets."""

    def __init__(self, reader):

        self.reader = reader

    @classmethod
    def load_builtin(cls, name='ml-100k'):
        """Load a built-in dataset.

        If the dataset has not already been loaded, it will be downloaded and
        saved. You will have to split your dataset using the :meth:`split
        <DatasetAutoFolds.split>` method. See an example in the :ref:`User
        Guide <cross_validate_example>`.

        Args:
            name(:obj:`string`): The name of the built-in dataset to load.
                Accepted values are 'ml-100k', 'ml-1m', and 'jester'.
                Default is 'ml-100k'.

        Returns:
            A :obj:`Dataset` object.

        Raises:
            ValueError: If the ``name`` parameter is incorrect.
        """

        try:
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name +
                             '. Accepted values are ' +
                             ', '.join(BUILTIN_DATASETS.keys()) + '.')

        # if dataset does not exist, offer to download it
        if not os.path.isfile(dataset.path):
            answered = False
            while not answered:
                print('Dataset ' + name + ' could not be found. Do you want '
                                          'to download it? [Y/n] ', end='')
                choice = input().lower()

                if choice in ['yes', 'y', '', 'omg this is so nice of you!!']:
                    answered = True
                elif choice in ['no', 'n', 'hell no why would i want that?!']:
                    answered = True
                    print("Ok then, I'm out!")
                    sys.exit()

            download_builtin_dataset(name)

        reader = Reader(**dataset.reader_params)

        return cls.load_from_file(file_path=dataset.path, reader=reader)

    @classmethod
    def load_from_file(cls, file_path, reader):
        """Load a dataset from a (custom) file.

        Use this if you want to use a custom dataset and all of the ratings are
        stored in one file. You will have to split your dataset using the
        :meth:`split <DatasetAutoFolds.split>` method. See an example in the
        :ref:`User Guide <load_from_file_example>`.


        Args:
            file_path(:obj:`string`): The path to the file containing ratings.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the file.
        """

        return DatasetAutoFolds(ratings_file=file_path, reader=reader)

    @classmethod
    def load_from_folds(cls, folds_files, reader):
        """Load a dataset where folds (for cross-validation) are predefined by
        some files.

        The purpose of this method is to cover a common use case where a
        dataset is already split into predefined folds, such as the
        movielens-100k dataset which defines files u1.base, u1.test, u2.base,
        u2.test, etc... It can also be used when you don't want to perform
        cross-validation but still want to specify your training and testing
        data (which comes down to 1-fold cross-validation anyway). See an
        example in the :ref:`User Guide <load_from_folds_example>`.


        Args:
            folds_files(:obj:`iterable` of :obj:`tuples`): The list of the
                folds. A fold is a tuple of the form ``(path_to_train_file,
                path_to_test_file)``.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the files.

        """

        return DatasetUserFolds(folds_files=folds_files, reader=reader)

    @classmethod
    def load_from_df(cls, df, reader):
        """Load a dataset from a pandas dataframe.

        Use this if you want to use a custom dataset that is stored in a pandas
        dataframe. See the :ref:`User Guide<load_from_df_example>` for an
        example.

        Args:
            df(`Dataframe`): The dataframe containing the ratings. It must have
                three columns, corresponding to the user (raw) ids, the item
                (raw) ids, and the ratings, in this order.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the file. Only the ``rating_scale`` field needs to be
                specified.
        """

        return DatasetAutoFolds(reader=reader, df=df)

    def read_ratings(self, file_name):
        """Return a list of ratings (user, item, rating, timestamp) read from
        file_name"""

        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_ratings

    def folds(self):
        """
        Generator function to iterate over the folds of the Dataset.

        .. warning::
            Deprecated since version 1.05. Use :ref:`cross-validation iterators
            <use_cross_validation_iterators>` instead. This method will be
            removed in later versions.

        Yields:
            tuple: :class:`Trainset <surprise.Trainset>` and testset
            of current fold.
        """

        warnings.warn('Using data.split() or using load_from_folds() '
                      'without using a CV iterator is now deprecated. ',
                      UserWarning)

        for raw_trainset, raw_testset in self.raw_folds():
            trainset = self.construct_trainset(raw_trainset)
            testset = self.construct_testset(raw_testset)
            yield trainset, testset

    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r, timestamp in raw_trainset:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(ur,
                            ir,
                            n_users,
                            n_items,
                            n_ratings,
                            self.reader.rating_scale,
                            self.reader.offset,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_trans)
                for (ruid, riid, r_ui_trans, _) in raw_testset]


class DatasetUserFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are predefined."""

    def __init__(self, folds_files=None, reader=None):

        Dataset.__init__(self, reader)
        self.folds_files = folds_files

        # check that all files actually exist.
        for train_test_files in self.folds_files:
            for f in train_test_files:
                if not os.path.isfile(os.path.expanduser(f)):
                    raise ValueError('File ' + str(f) + ' does not exist.')

    def raw_folds(self):
        for train_file, test_file in self.folds_files:
            raw_train_ratings = self.read_ratings(train_file)
            raw_test_ratings = self.read_ratings(test_file)
            yield raw_train_ratings, raw_test_ratings


class DatasetAutoFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are not predefined. (Or for when there are no folds at
    all)."""

    def __init__(self, ratings_file=None, reader=None, df=None):

        Dataset.__init__(self, reader)
        self.has_been_split = False  # flag indicating if split() was called.

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)
        elif df is not None:
            self.df = df
            self.raw_ratings = [(uid, iid, float(r) + self.reader.offset, None)
                                for (uid, iid, r) in
                                self.df.itertuples(index=False)]
        else:
            raise ValueError('Must specify ratings file or dataframe.')

    def build_full_trainset(self):
        """Do not split the dataset into folds and just return a trainset as
        is, built from the whole dataset.

        User can then query for predictions, as shown in the :ref:`User Guide
        <train_on_whole_trainset>`.

        Returns:
            The :class:`Trainset <surprise.Trainset>`.
        """

        return self.construct_trainset(self.raw_ratings)

    def raw_folds(self):

        if not self.has_been_split:
            self.split()

        def k_folds(seq, n_folds):
            """Inspired from scikit learn KFold method."""

            start, stop = 0, 0
            for fold_i in range(n_folds):
                start = stop
                stop += len(seq) // n_folds
                if fold_i < len(seq) % n_folds:
                    stop += 1
                yield seq[:start] + seq[stop:], seq[start:stop]

        return k_folds(self.raw_ratings, self.n_folds)

    def split(self, n_folds=5, shuffle=True):
        """
        Split the dataset into folds for future cross-validation.

        .. warning::
            Deprecated since version 1.05. Use :ref:`cross-validation iterators
            <use_cross_validation_iterators>` instead. This method will be
            removed in later versions.

        If you forget to call :meth:`split`, the dataset will be automatically
        shuffled and split for 5-fold cross-validation.

        You can obtain repeatable splits over your all your experiments by
        seeding the RNG: ::

            import random
            random.seed(my_seed)  # call this before you call split!

        Args:
            n_folds(:obj:`int`): The number of folds.
            shuffle(:obj:`bool`): Whether to shuffle ratings before splitting.
                If ``False``, folds will always be the same each time the
                experiment is run. Default is ``True``.
        """

        if n_folds > len(self.raw_ratings) or n_folds < 2:
            raise ValueError('Incorrect value for n_folds. Must be >=2 and '
                             'less than the number or entries')

        if shuffle:
            random.shuffle(self.raw_ratings)

        self.n_folds = n_folds
        self.has_been_split = True


class ImplicitDataset:

    def __init__(self, reader):

        self.reader = reader

    @classmethod
    def load_from_file(cls, file_path, reader):

        return ImplicitDatasetAutoFolds(ratings_file=file_path, reader=reader)

    @classmethod
    def load_from_df(cls, df, reader):

        return ImplicitDatasetAutoFolds(df=df, reader=reader)

    def read_ratings(self, file_name):

        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_ratings

    # TODO: deprecated, use 'construct_train_and_test'
    def folds(self):

        warnings.warn('Using data.split() or using load_from_folds() '
                      'without using a CV iterator is now deprecated. ',
                      UserWarning)

        for raw_trainset, raw_testset in self.raw_folds():
            trainset = self.construct_trainset(raw_trainset)
            testset = self.construct_testset(raw_testset)
            yield trainset, testset

    # TODO
    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r, timestamp in raw_trainset:

            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(ur,
                            ir,
                            n_users,
                            n_items,
                            n_ratings,
                            self.reader.rating_scale,
                            self.reader.offset,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_trans)
                for (ruid, riid, r_ui_trans, _) in raw_testset]

    # TODO, consider leave-one-out or leave-many-out, and sorting by timestamp
    def construct_train_and_test(self, leave_n_out=1, folds=5, order_by_time=False):

        if len(self.raw_ratings[0]) < 4 and order_by_time:
            print("timestamp is missing")
            order_by_time = False

        user_feedback = defaultdict(list)

        for u, i, r, *t in self.raw_ratings:
            user_feedback[u].append((i, r, t[0] if order_by_time else None))

        for __ in range(folds):
            raw_ratings_train = list()
            raw_ratings_test = list()

            for u in user_feedback.keys():
                u_ratings = random.shuffle(user_feedback[u])

                sorted_u_ratings = sorted(
                    u_ratings,
                    key=lambda x: (x[1], x[-1]) if order_by_time else lambda x: x[-1],
                    reverse=True
                )

                if sorted_u_ratings[leave_n_out] > 0:
                    raw_ratings_test.append([[u] + x for x in sorted_u_ratings[:leave_n_out]])
                    # TODO: add 0 rating or not ?
                    raw_ratings_train.append([[u] + x for x in sorted_u_ratings[leave_n_out:] if x[1] > 0])
                else:
                    # TODO: should users not in test set be put into train set?
                    print("Not enough positive samples")

            yield self.construct_trainset(raw_ratings_train), self.construct_testset(raw_ratings_test)

        pass


class ImplicitDatasetAutoFolds(ImplicitDataset):

    def __init__(self, ratings_file=None, df=None, reader=None):

        ImplicitDataset.__init__(self, reader)
        self.has_been_split = False  # flag indicating if split() was called.

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)
        elif df is not None:
            self.df = df
            self.raw_ratings = [
                (uid, iid, 1 if float(r) + self.offset > self.threshold else 0, None)
                for (uid, iid, r) in
                self.df.itertuples(index=False)
            ]
        else:
            raise ValueError('Must specify ratings file or dataframe.')

    def build_full_trainset(self):

        return self.construct_trainset(self.raw_ratings)

    # # TODO: deprecated
    # def raw_folds(self):
    #
    #     if not self.has_been_split:
    #         self.split()
    #
    #     def k_folds(seq, n_folds):
    #
    #         start, stop = 0, 0
    #         for fold_i in range(n_folds):
    #             start = stop
    #             stop += len(seq) // n_folds
    #             if fold_i < len(seq) % n_folds:
    #                 stop += 1
    #             yield seq[:start] + seq[stop:], seq[start:stop]
    #
    #     return k_folds(self.raw_ratings, self.n_folds)
    #
    # # TODO: deprecated
    # def split(self, n_folds=5, shuffle=True):
    #
    #     if n_folds > len(self.raw_ratings) or n_folds < 2:
    #         raise ValueError('Incorrect value for n_folds. Must be >=2 and '
    #                          'less than the number or entries')
    #
    #     if shuffle:
    #         random.shuffle(self.raw_ratings)
    #
    #     self.n_folds = n_folds
    #     self.has_been_split = True

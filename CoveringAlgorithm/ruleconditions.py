import numpy as np
from typing import List


class RuleConditions(object):
    """
    Class for binary rule condition
    """

    def __init__(self, features_name: List[str], features_index: List[int],
                 bmin: List[float], bmax: List[float], xmin: List[float],
                 xmax: List[float], values: List[float] = None):

        assert isinstance(features_name, (tuple, list, np.ndarray)), \
            'Type of parameter must be iterable tuple, list or array' % features_name
        self.features_name = features_name
        length = len(features_name)
        assert len(features_index) == length, \
            'Parameters must have the same length' % features_name
        self.features_index = features_index

        assert len(bmin) == length, \
            'Parameters must have the same length' % features_name
        assert len(bmax) == length, \
            'Parameters must have the same length' % features_name
        assert all(map(lambda a, b: a <= b, bmin, bmax)),\
            'Bmin must be smaller or equal than bmax (%s)' \
            % features_name
        self.bmin = bmin
        self.bmax = bmax

        assert len(xmax) == length, \
            'Parameters must have the same length' % features_name
        assert len(xmin) == length, \
            'Parameters must have the same length' % features_name
        self.xmin = xmin
        self.xmax = xmax

        if values is None:
            values = []

        self.values = [values]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        features = self.features_name
        return "Var: %s, Bmin: %s, Bmax: %s" % (features, self.bmin, self.bmax)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        to_hash = [(self.features_index[i], self.features_name[i],
                    self.bmin[i], self.bmax[i])
                   for i in range(len(self.features_index))]
        to_hash = frozenset(to_hash)
        return hash(to_hash)

    def transform(self, x: np.ndarray):
        """
        Transform a matrix xmat into an activation vector.
        It means an array of 0 and 1. 0 if the condition is not
        satisfied and 1 otherwise.

        Parameters
        ----------
        x: {array-like matrix, shape=(n_samples, n_features)}
              Input data

        Returns
        -------
        activation_vector: {array-like matrix, shape=(n_samples, 1)}
                     The activation vector
        """
        length = len(self.features_name)
        geq_min = True
        leq_min = True
        not_nan = True
        for i in range(length):
            col_index = self.features_index[i]
            x_col = x[:, col_index]

            # Turn x_col to array
            if len(x_col) > 1:
                x_col = np.squeeze(np.asarray(x_col))

            # if type(self.bmin[i]) == str:
            #     x_col = np.array(x_col, dtype=np.str)
            #
            #     temp = (x_col == self.bmin[i])
            #     temp |= (x_col == self.bmax[i])
            #     geq_min &= temp
            #     leq_min &= True
            #     not_nan &= True
            # else:
            x_col = np.array(x_col, dtype=np.float)

            # x_temp = [self.bmin[i] - 1 if x != x else x for x in x_col]
            geq_min &= np.greater_equal(x_col, self.bmin[i])

            # x_temp = [self.bmax[i] + 1 if x != x else x for x in x_col]
            leq_min &= np.less_equal(x_col, self.bmax[i])

            not_nan &= np.isfinite(x_col)

        activation_vector = 1 * (geq_min & leq_min & not_nan)

        return activation_vector

    """------   Getters   -----"""
    def get_param(self, param: str):
        """
        To get the parameter param
        """
        return getattr(self, param)

    def get_attr(self):
        """
        To get a list of attributes of self.
        It is useful to quickly create a RuleConditions
        from intersection of two rules
        """
        return [self.features_name,
                self.features_index,
                self.bmin, self.bmax,
                self.xmin, self.xmax]

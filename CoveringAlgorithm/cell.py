import numpy as np


class BaseCell:

    instances = []

    def __init__(self, activation: np.ndarray):
        self.activation = activation
        self.prediction = None

        if self not in BaseCell.instances:
            BaseCell.instances.append(self)

    def __eq__(self, other: "Cell"):
        return all(self.activation == other.activation)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.activation)

    def conditional_mean(self, y: np.ndarray):
        """Mean of all activated values

        If activation is None, we assume the given y have already been extracted from the activation vector,
        which saves time.
        """
        y_conditional = np.extract(self.activation, y)
        if len(y_conditional) > 0:
            self.prediction = float(np.nanmean(y_conditional))
        else:
            self.prediction = np.mean(y)


def Cell(*args, **kwargs):
    obj = BaseCell(*args, **kwargs)
    return BaseCell.instances[BaseCell.instances.index(obj)]
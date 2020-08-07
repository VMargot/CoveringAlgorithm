
import numpy as np
from sklearn.exceptions import NotFittedError


def check_is_fitted(estimator):
    if len(estimator.rules_list) == 0:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def mse_function(prediction_vector: np.ndarray, y: np.ndarray):
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)^2 $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vector = prediction_vector - y
    criterion = np.nanmean(error_vector ** 2)
    return criterion


def mae_function(prediction_vector: np.ndarray, y: np.ndarray):
    """
    Compute the mean absolute error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} |\\hat{y}_i - y_i| $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean absolute error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vect = np.abs(prediction_vector - y)
    criterion = np.nanmean(error_vect)
    return criterion


def aae_function(prediction_vector: np.ndarray, y: np.ndarray):
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)$"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vector = np.mean(np.abs(prediction_vector - y))
    median_error = np.mean(np.abs(y - np.median(y)))
    return error_vector / median_error


# def make_condition(rule: Rule):
#     """
#     Evaluate all suitable rules (i.e satisfying all criteria)
#     on a given feature.
#     Parameters
#     ----------
#     rule : {rule type}
#            A rule
#
#     Return
#     ------
#     conditions_str : {str type}
#                      A new string for the condition of the rule
#     """
#     conditions = rule.get_param('conditions').get_attr()
#     length = rule.get_param('length')
#     conditions_str = ''
#     for i in range(length):
#         if i > 0:
#             conditions_str += ' & '
#
#         conditions_str += conditions[0][i]
#         if conditions[2][i] == conditions[3][i]:
#             conditions_str += ' = '
#             conditions_str += str(conditions[2][i])
#         else:
#             conditions_str += r' $\in$ ['
#             conditions_str += str(conditions[2][i])
#             conditions_str += ', '
#             conditions_str += str(conditions[3][i])
#             conditions_str += ']'
#
#     return conditions_str


def calc_coverage(vect: np.ndarray):
    """
    Compute the coverage rate of an activation vector

    Parameters
    ----------
    vect : {array type}
           A activation vector. It means a sparse array with two
           different values 0, if the rule is not active
           and the 1 is the rule is active.

    Return
    ------
    cov : {float type}
          The coverage rate
    """
    u = np.sign(vect)
    return np.dot(u, u) / float(u.size)


def calc_prediction(activation_vector: np.ndarray, y: np.ndarray):
    """
    Compute the empirical conditional expectation of y
    knowing x

    Parameters
    ----------
    activation_vector : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    Return
    ------
    predictions : {float type}
           The empirical conditional expectation of y
           knowing x
    """
    y_cond = np.extract(activation_vector, y)
    if sum(~np.isnan(y_cond)) == 0:
        return 0
    else:
        predictions = np.nanmean(y_cond)
        return predictions


def calc_variance(activation_vector: np.ndarray, y: np.ndarray):
    """
    Compute the empirical conditional expectation of y
    knowing x

    Parameters
    ----------
    activation_vector : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    Return
    ------
    cond_var : {float type}
               The empirical conditional variance of y
               knowing x
    """
    # cov = calc_coverage(activation_vector)
    # y_cond = activation_vector * y
    # cond_var = 1. / cov * (np.mean(y_cond ** 2) - 1. / cov * np.mean(y_cond) ** 2)
    sub_y = np.extract(activation_vector, y)
    cond_var = np.var(sub_y)

    return cond_var


def calc_criterion(pred: float, y: np.ndarray, method: str = 'mse'):
    """
    Compute the criteria

    Parameters
    ----------
    pred : {float type}

    y : {array type}
        The real target values (real numbers)

    method : {string type}
             The method mse_function or mse_function criterion

    Return
    ------
    criterion : {float type}
           Criteria value
    """
    prediction_vector = pred * y.astype('bool')

    if method == 'mse':
        criterion = mse_function(prediction_vector, y)

    elif method == 'mae':
        criterion = mae_function(prediction_vector, y)

    elif method == 'aae':
        criterion = aae_function(prediction_vector, y)

    else:
        raise 'Method %s unknown' % method

    return criterion


def dist(u: np.ndarray, v: np.ndarray):
    """
    Compute the distance between two prediction vector

    Parameters
    ----------
    u,v : {array type}
          A predictor vector. It means a sparse array with two
          different values 0, if the rule is not active
          and the prediction is the rule is active.

    Return
    ------
    Distance between u and v
    """
    assert len(u) == len(v), \
        'The two array must have the same length'
    u = np.sign(u)
    v = np.sign(v)
    num = np.dot(u, v)
    deno = min(np.dot(u, u),
               np.dot(v, v))
    return 1 - num / deno

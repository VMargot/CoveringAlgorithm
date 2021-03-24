from typing import List
import copy
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from rule.rule import Rule
from ruleset.ruleset import RuleSet
from condition.hyperrectanglecondition import HyperrectangleCondition


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


def calc_conditional_mean(activation_vector: np.ndarray, y: np.ndarray):
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


def extract_rules_rulefit(rules_df: pd.DataFrame,
                          features_names: List[str],
                          bmins_list: List[float],
                          bmaxs_list: List[float]) -> RuleSet:
    rulefit_ruleset = RuleSet()

    for rule in rules_df['rule'].values:
        if '&' in rule:
            rule_split = rule.split(' & ')
        else:
            rule_split = [rule]

        features_name = []
        features_index = []
        bmin = []
        bmax = []
        xmax = []
        xmin = []

        for sub_rule in rule_split:
            sub_rule = sub_rule.replace('=', '')

            if '>' in sub_rule:
                sub_rule = sub_rule.split(' > ')
                if 'feature_' in sub_rule[0]:
                    feat_id = sub_rule[0].split('_')[-1]
                    feat_id = int(feat_id)
                    features_name += [features_names[feat_id]]
                else:
                    features_name += [sub_rule[0]]
                    feat_id = features_names.index(sub_rule[0])
                features_index += [feat_id]
                bmin += [float(sub_rule[-1])]
                bmax += [bmaxs_list[feat_id]]
            else:
                sub_rule = sub_rule.split(' < ')
                if 'feature_' in sub_rule[0]:
                    feat_id = sub_rule[0].split('_')[-1]
                    feat_id = int(feat_id)
                    features_name += [features_names[feat_id]]
                else:
                    features_name += [sub_rule[0]]
                    feat_id = features_names.index(sub_rule[0])
                features_index += [feat_id]
                bmax += [float(sub_rule[-1])]
                bmin += [bmins_list[feat_id]]

            xmax += [bmaxs_list[feat_id]]
            xmin += [bmins_list[feat_id]]

        new_cond = HyperrectangleCondition(features_names=features_name,
                                           features_indexes=features_index,
                                           bmins=bmin, bmaxs=bmax)
        new_rg = Rule(copy.deepcopy(new_cond))
        rulefit_ruleset += new_rg

    return rulefit_ruleset


def make_rs_from_r(df: pd.DataFrame, features_list: List[str], xmin: List[float], xmax: List[float]) -> RuleSet:
    rules = df['Rules'].values
    r_ruleset = RuleSet()
    for i in range(len(rules)):
        rl_i = rules[i].split(' AND ')
        cp = len(rl_i)
        conditions = [[] for _ in range(6)]

        for j in range(cp):
            feature_name = rl_i[j].split(' in ')[0]
            feature_name = feature_name.replace('.', ' ')
            feature_id = features_list.index(feature_name)
            bmin = rl_i[j].split(' in ')[1].split(';')[0]
            if bmin == '-Inf':
                bmin = xmin[feature_id]
            else:
                bmin = float(bmin)
            bmax = rl_i[j].split(' in ')[1].split(';')[1]
            if bmax == 'Inf':
                bmax = xmax[feature_id]
            else:
                bmax = float(bmax)

            conditions[0] += [feature_name]
            conditions[1] += [feature_id]
            conditions[2] += [bmin]
            conditions[3] += [bmax]
            conditions[4] += [xmin[feature_id]]
            conditions[5] += [xmax[feature_id]]

        new_cond = HyperrectangleCondition(features_indexes=conditions[1], bmins=conditions[2], bmaxs=conditions[3],
                                           features_names=conditions[0])
        r_ruleset += Rule(new_cond)

    return r_ruleset


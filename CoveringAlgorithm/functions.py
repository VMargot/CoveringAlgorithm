from typing import List
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from ruleskit import RegressionRule
from ruleskit import RuleSet
from ruleskit import HyperrectangleCondition


def check_is_fitted(estimator):
    if len(estimator.rules_list) == 0:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )
        raise NotFittedError(msg % {"name": type(estimator).__name__})


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
    assert len(prediction_vector) == len(y), "The two array must have the same length"
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
    assert len(prediction_vector) == len(y), "The two array must have the same length"
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
    assert len(prediction_vector) == len(y), "The two array must have the same length"
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
    assert len(u) == len(v), "The two array must have the same length"
    u = np.sign(u)
    v = np.sign(v)
    num = np.dot(u, v)
    deno = min(np.dot(u, u), np.dot(v, v))
    return 1 - num / deno


def extract_rules_rulefit(
    rules_df: pd.DataFrame,
    features_names_list: List[str],
    bmins_list: List[float],
    bmaxs_list: List[float],
) -> RuleSet:
    rulefit_ruleset = RuleSet()

    for rule in rules_df["rule"].values:
        if "&" in rule:
            rule_split = rule.split(" & ")
        else:
            rule_split = [rule]

        features_names = []
        features_indexes = []
        bmins = []
        bmaxs = []

        for sub_rule in rule_split:
            sub_rule = sub_rule.replace("=", "")

            if ">" in sub_rule:
                sub_rule = sub_rule.split(" > ")
                if "feature_" in sub_rule[0]:
                    feat_id = sub_rule[0].split("_")[-1]
                    feat_id = int(feat_id)
                    features_names += [features_names_list[feat_id]]
                else:
                    features_names += [sub_rule[0]]
                    feat_id = features_names_list.index(sub_rule[0])
                features_indexes += [feat_id]
                bmins += [float(sub_rule[-1])]
                bmaxs += [bmaxs_list[feat_id]]
            else:
                sub_rule = sub_rule.split(" < ")
                if "feature_" in sub_rule[0]:
                    feat_id = sub_rule[0].split("_")[-1]
                    feat_id = int(feat_id)
                    features_names += [features_names_list[feat_id]]
                else:
                    features_names += [sub_rule[0]]
                    feat_id = features_names_list.index(sub_rule[0])
                features_indexes += [feat_id]
                bmaxs += [float(sub_rule[-1])]
                bmins += [bmins_list[feat_id]]

        new_cond = HyperrectangleCondition(
            features_names=features_names,
            features_indexes=features_indexes,
            bmins=bmins,
            bmaxs=bmaxs,
        )
        rulefit_ruleset += RegressionRule(new_cond)

    return rulefit_ruleset


def make_rs_from_r(
    df: pd.DataFrame, features_list: List[str], xmin: List[float], xmax: List[float]
) -> RuleSet:
    rules = df["Rules"].values
    r_ruleset = RuleSet()
    for i in range(len(rules)):
        rl_i = rules[i].split(" AND ")
        cp = len(rl_i)
        conditions = [[] for _ in range(6)]

        for j in range(cp):
            feature_name = rl_i[j].split(" in ")[0]
            feature_name = feature_name.replace(".", " ")
            feature_id = features_list.index(feature_name)
            bmin = rl_i[j].split(" in ")[1].split(";")[0]
            if bmin == "-Inf":
                bmin = xmin[feature_id]
            else:
                bmin = float(bmin)
            bmax = rl_i[j].split(" in ")[1].split(";")[1]
            if bmax == "Inf":
                bmax = xmax[feature_id]
            else:
                bmax = float(bmax)

            conditions[0] += [feature_id]
            conditions[1] += [bmin]
            conditions[2] += [bmax]
            conditions[3] += [feature_name]

        new_cond = HyperrectangleCondition(
            features_indexes=conditions[0],
            bmins=conditions[1],
            bmaxs=conditions[2],
            features_names=conditions[3],
        )
        r_ruleset += RegressionRule(new_cond)

    return r_ruleset

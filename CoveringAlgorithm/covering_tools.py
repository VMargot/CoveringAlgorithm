import copy
import math
from typing import List, Union, Tuple
import numpy as np

from CoveringAlgorithm.functions import calc_conditional_mean
from ruleset.ruleset import RuleSet
from rule.rule import Rule
from rule.activation import Activation


def interpretability_index(rs: Union[RuleSet, List[Rule]]) -> int:
    return sum(map(lambda r: len(r), rs))


def union_test(rule: Rule, act: Activation, gamma=0.80):
    """
    Test to know if a rule (self) and an activation vector have
    at more gamma percent of points in common
    """
    rule_activation = rule._activation
    intersect_vect = rule_activation & act

    pts_inter = intersect_vect.sum_ones()
    pts_act = act.sum_ones()
    pts_rule = rule_activation.sum_ones()

    ans = (pts_inter < gamma * pts_rule) and (pts_inter < gamma * pts_act)

    return ans


def select_rules(rules_list: List[Rule],
                 gamma: float = 1.0,
                 selected_rs: RuleSet = None) -> RuleSet:
    """
    Returns a subset of a given rs. This subset is seeking by
    minimization/maximization of the criterion on the training set
    """
    # Then optimization
    if selected_rs is None or len(selected_rs) == 0:
        selected_rs = RuleSet(rules_list[:1])
        id_rule = 1
    else:
        id_rule = 0

    nb_rules = len(rules_list)

    for i in range(id_rule, nb_rules):
        if selected_rs.calc_coverage_rate() == 1.0:
            break
        rs_copy = copy.deepcopy(selected_rs)
        new_rules = rules_list[i]
        # Test union criteria for each rule in the current selected RuleSet
        utest = [union_test(new_rules, rule._activation, gamma) for rule in rs_copy]
        if all(utest) and union_test(new_rules, selected_rs.get_activation(), gamma):
            selected_rs += new_rules
    return selected_rs


def get_significant(rules_list, ymean, beta, gamma, sigma2) -> Tuple[RuleSet, List[Rule]]:
    def is_significant(rule, beta, ymean, sigma2):
        return beta * abs(ymean - rule.prediction) >= math.sqrt(max(0, rule.std**2 - sigma2))

    filtered_rules = filter(lambda rule: is_significant(rule, beta, ymean, sigma2),
                            rules_list)

    significant_rules = list(filtered_rules)
    [setattr(rule, 'significant', True) for rule in significant_rules]

    if len(significant_rules) > 0:
        significant_rules = sorted(significant_rules, key=lambda x: x.coverage,
                                   reverse=True)
        # significant_rs.sort_by(crit='crit', maximized=False)
        significant_selected_rs = select_rules(rules_list=significant_rules, gamma=gamma)
    else:
        significant_selected_rs = RuleSet()

    return significant_selected_rs, significant_rules


def add_insignificant_rules(rules_list, rs, epsilon, sigma2, gamma):
    def is_insignificant(rule, epsilon, sigma2):
        return epsilon >= math.sqrt(max(0, rule.std**2 - sigma2))

    insignificant_rules = filter(lambda rule: is_insignificant(rule, epsilon, sigma2),
                                 rules_list)
    insignificant_rules = list(insignificant_rules)
    [setattr(rule, 'significant', False) for rule in insignificant_rules]

    if len(insignificant_rules) > 0:
        insignificant_rs = sorted(insignificant_rules, key=lambda x: x.std,
                                  reverse=False)
        selected_rs = select_rules(rules_list=insignificant_rs, gamma=gamma,
                                   selected_rs=rs)
    else:
        selected_rs = RuleSet()

    return selected_rs


def find_covering(rules_list: List[Rule], y: np.ndarray, sigma2: float = None,
                  alpha: float = 1. / 2 - 1 / 100, gamma: float = 0.95) -> RuleSet:

    n_train = len(y)
    cov_min = n_train ** (-alpha)
    # print('Minimal coverage rate:', cov_min)

    sub_rules_list = list(filter(lambda rule: rule.coverage > cov_min, rules_list))
    # print('Nb of rules with good coverage rate:', len(sub_rules_list))

    if sigma2 is None:
        var_list = [rg.std**2 for rg in sub_rules_list]
        sigma2 = min(list(filter(lambda v: v > 0, var_list)))
        # print('Sigma 2 estimation', sigma2)

    beta = pow(n_train, alpha / 2. - 1. / 4)
    epsilon = beta * np.std(y)

    significant_selected_rs, significant_rules = get_significant(sub_rules_list, np.mean(y),
                                                                 beta, gamma, sigma2)

    if significant_selected_rs.calc_coverage_rate() < 1.0:
        sub_rules_list = list(filter(lambda r: r not in significant_rules, sub_rules_list))
        selected_rs = add_insignificant_rules(sub_rules_list, significant_selected_rs,
                                              epsilon, sigma2, gamma)
    else:
        selected_rs = significant_selected_rs

    return selected_rs


def calc_prediction(rules_list: RuleSet, ytrain: np.ndarray, x: np.ndarray):
    """
    Computes the prediction vector
    using an rule based partition
    """
    # Activation of all rules in the learning set
    activation_matrix = [rule.activation for rule in rules_list]
    activation_matrix = np.array(activation_matrix)

    prediction_matrix = [rule.calc_activation(x).get_array() for rule in rules_list]
    prediction_matrix = np.array(prediction_matrix).T

    no_activation_matrix = np.logical_not(prediction_matrix)

    nb_rules_active = prediction_matrix.sum(axis=1)
    nb_rules_active[nb_rules_active == 0] = -1  # If no rule is activated

    # Activation of the intersection of all NOT activated rules at each row
    no_activation_vector = np.dot(no_activation_matrix, activation_matrix)
    no_activation_vector = np.array(no_activation_vector, dtype='int')

    # Activation of the intersection of all activated rules at each row
    dot_activation = np.dot(prediction_matrix, activation_matrix)
    dot_activation = np.array([np.equal(act, nb_rules) for act, nb_rules in
                               zip(dot_activation, nb_rules_active)], dtype='int')

    # Calculation of the binary vector for cells of the partition et each row
    cells = ((dot_activation - no_activation_vector) > 0)

    # Calculation of the conditional expectation in each cell
    prediction_vector = [calc_conditional_mean(act, ytrain) if sum(act) > 0 else
                         np.mean(ytrain) for act in cells]
    prediction_vector = np.array(prediction_vector)
    return prediction_vector




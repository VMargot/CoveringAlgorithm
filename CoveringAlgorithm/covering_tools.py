import copy
import math
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import _tree
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches

from CoveringAlgorithm import functions as f
from CoveringAlgorithm.ruleset import RuleSet
from CoveringAlgorithm.rule import Rule
from CoveringAlgorithm.ruleconditions import RuleConditions


def extract_rules_from_tree(tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
                            features: List[str],
                            xmin: List[float],
                            xmax: List[float],
                            get_leaf: bool = False) -> List[Rule]:
    dt = tree.tree_

    def visitor(node, depth, cond=None, rule_list=None):
        if rule_list is None:
            rule_list = []
        if dt.feature[node] != _tree.TREE_UNDEFINED:
            # If
            new_cond = RuleConditions([features[dt.feature[node]]],
                                      [dt.feature[node]],
                                      bmin=[xmin[dt.feature[node]]],
                                      bmax=[dt.threshold[node]],
                                      xmin=[xmin[dt.feature[node]]],
                                      xmax=[xmax[dt.feature[node]]])
            if cond is not None:
                if dt.feature[node] not in cond.features_index:
                    conditions_list = list(map(lambda c1, c2: c1 + c2, cond.get_attr(),
                                               new_cond.get_attr()))
                
                    new_cond = RuleConditions(features_name=conditions_list[0],
                                              features_index=conditions_list[1],
                                              bmin=conditions_list[2],
                                              bmax=conditions_list[3],
                                              xmax=conditions_list[5],
                                              xmin=conditions_list[4])
                else:
                    new_bmax = dt.threshold[node]
                    new_cond = copy.deepcopy(cond)
                    place = cond.features_index.index(dt.feature[node])
                    new_cond.bmax[place] = min(new_bmax, new_cond.bmax[place])
        
            # print (Rule(new_cond))
            new_rg = Rule(copy.deepcopy(new_cond))
            if get_leaf is False:
                rule_list.append(new_rg)
        
            rule_list = visitor(dt.children_left[node], depth + 1,
                                new_cond, rule_list)
        
            # Else
            new_cond = RuleConditions([features[dt.feature[node]]],
                                      [dt.feature[node]],
                                      bmin=[dt.threshold[node]],
                                      bmax=[xmax[dt.feature[node]]],
                                      xmin=[xmin[dt.feature[node]]],
                                      xmax=[xmax[dt.feature[node]]])
            if cond is not None:
                if dt.feature[node] not in cond.features_index:
                    conditions_list = list(map(lambda c1, c2: c1 + c2, cond.get_attr(),
                                               new_cond.get_attr()))
                    new_cond = RuleConditions(features_name=conditions_list[0],
                                              features_index=conditions_list[1],
                                              bmin=conditions_list[2],
                                              bmax=conditions_list[3],
                                              xmax=conditions_list[5],
                                              xmin=conditions_list[4])
                else:
                    new_bmin = dt.threshold[node]
                    new_bmax = xmax[dt.feature[node]]
                    new_cond = copy.deepcopy(cond)
                    place = new_cond.features_index.index(dt.feature[node])
                    new_cond.bmin[place] = max(new_bmin, new_cond.bmin[place])
                    new_cond.bmax[place] = max(new_bmax, new_cond.bmax[place])
        
            new_rg = Rule(copy.deepcopy(new_cond))
            if get_leaf is False:
                rule_list.append(new_rg)
        
            rule_list = visitor(dt.children_right[node], depth + 1, new_cond, rule_list)

        elif get_leaf:
            rule_list.append(Rule(copy.deepcopy(cond)))

        return rule_list

    rule_list = visitor(0, 1)
    return rule_list


def extract_rules_rulefit(rules: pd.DataFrame,
                          features: List[str],
                          bmin_list: List[float],
                          bmax_list: List[float]) -> List[Rule]:
    rule_list = []

    for rule in rules['rule'].values:
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
                    features_name += [features[feat_id]]
                else:
                    features_name += [sub_rule[0]]
                    feat_id = features.index(sub_rule[0])
                features_index += [feat_id]
                bmin += [float(sub_rule[-1])]
                bmax += [bmax_list[feat_id]]
            else:
                sub_rule = sub_rule.split(' < ')
                if 'feature_' in sub_rule[0]:
                    feat_id = sub_rule[0].split('_')[-1]
                    feat_id = int(feat_id)
                    features_name += [features[feat_id]]
                else:
                    features_name += [sub_rule[0]]
                    feat_id = features.index(sub_rule[0])
                features_index += [feat_id]
                bmax += [float(sub_rule[-1])]
                bmin += [bmin_list[feat_id]]

            xmax += [bmax_list[feat_id]]
            xmin += [bmin_list[feat_id]]

        new_cond = RuleConditions(features_name=features_name,
                                  features_index=features_index,
                                  bmin=bmin, bmax=bmax,
                                  xmin=xmin, xmax=xmax)
        new_rg = Rule(copy.deepcopy(new_cond))
        rule_list.append(new_rg)

    return rule_list


def select_rs(rs: Union[List[Rule], RuleSet],
              gamma: float = 1.0,
              selected_rs: RuleSet = None) -> RuleSet:
    """
    Returns a subset of a given rs. This subset is seeking by
    minimization/maximization of the criterion on the training set
    """
    # Then optimization
    if selected_rs is None or len(selected_rs) == 0:
        selected_rs = RuleSet(rs[:1])
        id_rule = 1
    else:
        id_rule = 0

    nb_rules = len(rs)

    for i in range(id_rule, nb_rules):
        rs_copy = copy.deepcopy(selected_rs)
        new_rules = rs[i]
        # Test union criteria for each rule in the current selected RuleSet
        utest = [new_rules.union_test(rule.get_activation(), gamma) for rule in rs_copy]
        if all(utest) and new_rules.union_test(selected_rs.calc_activation(), gamma):
            selected_rs.append(new_rules)

    return selected_rs


def get_significant(rules_list, ymean, beta, gamma, sigma2):
    def is_significant(rule, beta, ymean, sigma2):
        return beta * abs(ymean - rule.pred) >= math.sqrt(max(0, rule.var - sigma2))

    filtered_rules = filter(lambda rule: is_significant(rule, beta, ymean, sigma2),
                            rules_list)

    significant_rules = list(filtered_rules)
    [rule.set_params(significant=True) for rule in significant_rules]
    significant_rs = RuleSet(significant_rules)

    if len(significant_rs) > 0:
        significant_rs.sort_by(crit='cov', maximized=True)
        # significant_rs.sort_by(crit='crit', maximized=False)
        significant_selected_rs = select_rs(rs=significant_rs, gamma=gamma)
    else:
        significant_selected_rs = RuleSet([])

    return significant_selected_rs


def add_insignificant_rules(rules_list, rs, epsilon, sigma2, gamma):
    def is_significant(rule, epsilon, sigma2):
        return epsilon >= math.sqrt(max(0, rule.var - sigma2))

    insignificant_rule = filter(lambda rule: is_significant(rule, epsilon, sigma2),
                                rules_list)
    insignificant_rule = list(insignificant_rule)
    insignificant_rs = RuleSet(insignificant_rule)
    [rule.set_params(significant=False) for rule in insignificant_rs]

    if len(insignificant_rs) > 0:
        insignificant_rs.sort_by(crit='var', maximized=False)
        selected_rs = select_rs(rs=insignificant_rs, gamma=gamma,
                                selected_rs=rs)
    else:
        selected_rs = RuleSet([])

    return selected_rs


def find_covering(rules_list, y, sigma2=None,
                  alpha=1. / 2 - 1 / 100,
                  gamma=0.95):

    n_train = len(y)
    cov_min = n_train ** (-alpha)
    # print('Minimal coverage rate:', cov_min)

    sub_rules_list = list(filter(lambda rule: rule.cov > cov_min, rules_list))
    # print('Nb of rules with good coverage rate:', len(sub_rules_list))

    if sigma2 is None:
        var_list = [rg.var for rg in sub_rules_list]
        sigma2 = min(list(filter(lambda v: v > 0, var_list)))
        # print('Sigma 2 estimation', sigma2)

    beta = pow(n_train, alpha / 2. - 1. / 4)
    epsilon = beta * np.std(y)

    significant_selected_rs = get_significant(sub_rules_list, np.mean(y),
                                              beta, gamma, sigma2)

    if significant_selected_rs.calc_coverage() < 1.0:
        selected_rs = add_insignificant_rules(sub_rules_list, significant_selected_rs,
                                              epsilon, sigma2, gamma)
    else:
        selected_rs = significant_selected_rs

    return selected_rs


def calc_pred(ruleset, ytrain, x):
    """
    Computes the prediction vector
    using an rule based partition
    """
    # Activation of all rules in the learning set
    activation_matrix = [rule.get_activation() for rule in ruleset]
    activation_matrix = np.array(activation_matrix)

    prediction_matrix = [rule.calc_activation(x) for rule in ruleset]
    prediction_matrix = np.array(prediction_matrix).T

    no_activation_matrix = np.logical_not(prediction_matrix)

    nb_rules_active = prediction_matrix.sum(axis=1)
    nb_rules_active[nb_rules_active == 0] = -1  # If no rule is activated

    # Activation of the intersection of all NOT activated rules at each row
    no_activation_vector = np.dot(no_activation_matrix, activation_matrix)
    no_activation_vector = np.array(no_activation_vector,
                                    dtype='int')

    dot_activation = np.dot(prediction_matrix, activation_matrix)
    dot_activation = np.array([np.equal(act, nb_rules) for act, nb_rules in
                               zip(dot_activation, nb_rules_active)], dtype='int')

    # Calculation of the binary vector for cells of the partition et each row
    cells = ((dot_activation - no_activation_vector) > 0)

    # Calculation of the conditional expectation in each cell
    prediction_vector = [f.calc_prediction(act, ytrain) for act in cells]
    prediction_vector = np.array(prediction_vector)
    prediction_vector[prediction_vector == 0] = np.mean(ytrain)
    return prediction_vector


def make_condition(rule):
    """
    Evaluate all suitable rules (i.e satisfying all criteria)
    on a given feature.
    Parameters
    ----------
    rule: {rule type}
           A rule

    Return
    ------
    conditions_str: {str type}
                     A new string for the condition of the rule
    """
    conditions = rule.get_param('conditions').get_attr()
    length = rule.get_param('length')

    conditions_str = ''
    for i in range(length):
        if i > 0:
            conditions_str += ' & '
    
        conditions_str += conditions[0][i]
        if conditions[2][i] == round(conditions[3][i], 2):
            conditions_str += ' = '
            conditions_str += str(round(conditions[2][i], 2))
        else:
            conditions_str += r' $\in$ ['
            conditions_str += str(round(conditions[2][i], 2))
            conditions_str += ', '
            conditions_str += str(round(conditions[3][i], 2))
            conditions_str += ']'

    return conditions_str


def make_selected_df(rs):
    """
    Returns
    -------
    selected_df: {DataFrame type}
                  DataFrame of selected RuleSet for presentation
    """
    df = rs.to_df()

    df.rename(columns={"Cov": "Coverage", "Pred": "Prediction",
                       'Var': 'Variance', 'Crit': 'Criterium'},
              inplace=True)

    df['Conditions'] = [make_condition(rule) for rule in rs]
    selected_df = df[['Conditions', 'Coverage',
                      'Prediction', 'Variance',
                      'Criterium']].copy()

    selected_df['Coverage'] = selected_df.Coverage.round(2)
    selected_df['Prediction'] = selected_df.Prediction.round(2)
    selected_df['Variance'] = selected_df.Variance.round(2)
    selected_df['Criterium'] = selected_df.Criterium.round(2)

    return selected_df


def plot_rules(selected_rs, ymax, ymin,
               xmax, xmin, var1, var2,
               cm=plt.cm.RdBu, cp=None):
    """
    Plot the rectangle activation zone of rules in a 2D plot
    the color is corresponding to the intensity of the prediction
    Parameters
    ----------
    selected_rs: {RuleSet type}
                  The set of selcted rules
    ymax: {float or int type}
          The maximal value of the variable Y

    ymin: {float or int type}
          The minimal value of the variable Y

    xmax: {tuple or list type of length 2}
          The 2 maximal values of var1 and var2

    xmin: {tuple or list type of length 2}
          The 2 minimal values of var1 and var2

    var1: {string type}
           Name of the first variable

    var2: {string type}
           Name of the second variable
    cm: {}
    cp: {int type}, optional
         Option to plot only the cp1 or cp2 rules

    Returns
    -------
    Draw the graphic
    """

    nb_color = cm.N
    selected_rs.sort_by(crit='cov', maximized=True)
    if cp is not None:
        sub_ruleset = selected_rs.extract_cp(cp)
    else:
        sub_ruleset = selected_rs

    plt.plot()

    for rg in sub_ruleset:
        rg_condition = rg.conditions
    
        var = rg_condition.get_param('features_index')
        bmin = rg_condition.get_param('bmin')
        bmax = rg_condition.get_param('bmax')
    
        cp_rg = rg.get_param('length')
    
        if rg.get_param('pred') > 0:
            hatch = '/'
            alpha = (rg.get_param('pred') / ymax)
            idx = int(nb_color / 2 + alpha * nb_color / 2) + 1
            facecolor = matplotlib.colors.rgb2hex(cm(idx))
        else:
            hatch = '\\'
            alpha = (rg.get_param('pred') / ymin)
            idx = int(nb_color / 2 - alpha * nb_color / 2) + 1
            facecolor = matplotlib.colors.rgb2hex(cm(idx))
    
        if cp_rg == 1:
            if var[0] == var1:
                p = patches.Rectangle((bmin[0], xmin[1]),  # origin
                                      abs(bmax[0] - bmin[0]),  # width
                                      xmax[1] - xmin[1],  # height
                                      hatch=hatch, facecolor=facecolor,
                                      alpha=alpha)
                plt.gca().add_patch(p)
        
            elif var[0] == var2:
                p = patches.Rectangle((xmin[0], bmin[0]),
                                      xmax[0] - xmin[0],
                                      abs(bmax[0] - bmin[0]),
                                      hatch=hatch, facecolor=facecolor,
                                      alpha=alpha)
                plt.gca().add_patch(p)
    
        elif cp_rg == 2:
            if var[0] == var1 and var[1] == var2:
                p = patches.Rectangle((bmin[0], bmin[1]),
                                      abs(bmax[0] - bmin[0]),
                                      abs(bmax[1] - bmin[1]),
                                      hatch=hatch, facecolor=facecolor,
                                      alpha=alpha)
                plt.gca().add_patch(p)
        
            elif var[1] == var1 and var[0] == var2:
                p = patches.Rectangle((bmin[1], bmin[0]),
                                      abs(bmax[1] - bmin[1]),
                                      abs(bmax[0] - bmin[0]),
                                      hatch=hatch, facecolor=facecolor,
                                      alpha=alpha)
                plt.gca().add_patch(p)

    if cp is None:
        plt.gca().set_title('Rules covering', fontsize=25)
    else:
        plt.gca().set_title('Rules cp%s covering' % str(cp), fontsize=25)

    # plt.colorbar()
    plt.gca().axis([xmin[0], xmax[0], xmin[1], xmax[1]])

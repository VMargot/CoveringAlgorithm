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
            
            # print (Rule(new_cond))
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
    
    # for i in tqdm.tqdm(range(id_rule, nb_rules), desc='Selection'):
    for i in range(id_rule, nb_rules):
        rs_copy = copy.deepcopy(selected_rs)
        new_rules = rs[i]
        
        utest = [new_rules.union_test(rule.get_activation(),
                                      gamma)
                 for rule in rs_copy]
        
        if all(utest) and new_rules.union_test(selected_rs.calc_activation(),
                                               gamma):
            new_rs = copy.deepcopy(selected_rs)
            new_rs.append(new_rules)
            
            selected_rs = copy.deepcopy(new_rs)
    
    return selected_rs


def get_significant(rules_list, ymean, beta, gamma, sigma2):
    def is_significant(rule, beta, ymean, sigma2):
        return beta * abs(ymean - rule.pred) >= math.sqrt(max(0, rule.var - sigma2))
    
    filtered_rules = filter(lambda rule: is_significant(rule, beta, ymean, sigma2),
                            rules_list)
    
    significant_rules = list(filtered_rules)
    [rule.set_params(significant=True) for rule in significant_rules]
    
    # print('Nb of significant rules', len(significant_rules))
    significant_rs = RuleSet(significant_rules)
    # print('Coverage rate of significant rule:', significant_rs.calc_coverage())
    
    significant_rs.sort_by(crit='cov', maximized=True)
    if len(significant_rs) > 0:
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
    # print('Nb of insignificant rules', len(insignificant_rule))
    insignificant_rs = RuleSet(insignificant_rule)
    # print('Coverage rate of significant rule:', insignificant_rs.calc_coverage())
    [rule.set_params(significant=False) for rule in insignificant_rs]
    
    if len(insignificant_rs) > 0:
        insignificant_rs.sort_by(crit='var', maximized=False)
        selected_rs = select_rs(rs=insignificant_rs, gamma=gamma,
                                selected_rs=rs)
    else:
        selected_rs = RuleSet([])
    
    # print('Number of rules:', len(selected_rs))
    # print('Coverage rate of the selected RuleSet ', selected_rs.calc_coverage())
    
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
    # print('Beta coefficient:', beta)
    epsilon = beta * np.std(y)
    # print('Epsilon coefficient:', epsilon)
    
    significant_selected_rs = get_significant(sub_rules_list, np.mean(y),
                                              beta, gamma, sigma2)
    
    if significant_selected_rs.calc_coverage() < 1.0:
        selected_rs = add_insignificant_rules(sub_rules_list, significant_selected_rs,
                                              epsilon, sigma2, gamma)
        
        # if selected_rs.calc_coverage() < 1.0:
        #     new_rs = extend_bounds(selected_rs, x, y, np.mean(y),
        #                            sigma2, beta, epsilon)
        # else:
        #     # print('No norule added')
        #     new_rs = copy.copy(selected_rs)
    else:
        # print('Significant rules form a covering')
        selected_rs = significant_selected_rs
        # new_rs = copy.copy(significant_selected_rs)
    
    return selected_rs


def extend_bounds(rs, x, y, ymean, sigma2, beta, epsilon):
    def dist(low, up, x):
        if x < low:
            return low - x
        elif x > up:
            return x - up
        else:
            return 0
    
    def calc_dist(rule, x):
        aa = rule.conditions.features_index
        bmin = rule.conditions.bmin
        bmax = rule.conditions.bmax
        return [dist(low, up, z) for low, up, z in zip(bmin, bmax, x[aa])]

    no_act = 1 - rs.calc_activation(x)
    ids = np.where(no_act == 1)[0]
    temp = [[calc_dist(rule, x) for rule in rs] for x in x[ids, :]]
    temp2 = [[sum(t) for t in temp[i]] for i in range(len(temp))]

    t = np.zeros(len(temp[0]))
    for i in range(len(temp)):
        t += temp2[i]
    t = list(t)

    id_rule = t.index(min(t))
    tt = [temp[i][id_rule] for i in range(len(temp))]
    val_change = list(map(max, (zip(*tt))))

    new_rule = copy.deepcopy(rs[id_rule])
    old_bmin = new_rule.conditions.bmin
    old_bmax = new_rule.conditions.bmax
    bmin = []
    bmax = []

    j = 0
    for i in new_rule.conditions.features_index:
        if val_change[j] == 0:
            bmin.append(old_bmin[j])
            bmax.append(old_bmax[j])
        else:
            sub_x = x[ids, i]
            x_max = max(sub_x)
            x_min = min(sub_x)
        
            if x_min < old_bmin[j]:
                if x_max <= old_bmax[j]:
                    bmin.append(old_bmin[j] - val_change[j])
                    bmax.append(old_bmax[j])
                elif x_min - old_bmin[j] > old_bmax[j] - x_max:
                    bmin.append(old_bmin[j] - val_change[j])
                    bmax.append(old_bmax[j])
                else:
                    bmax.append(old_bmax[j] + val_change[j])
                    bmin.append(old_bmin[j])
            else:
                bmax.append(old_bmax[j] + val_change[j])
                bmin.append(old_bmin[j])
        j += 1

    new_rule.conditions.bmin = bmin
    new_rule.conditions.bmax = bmax
    
    new_rule.calc_stats(y=y, x=x, cov_min=0.0, cov_max=1.1)
    
    left_term = beta * abs(ymean - new_rule.pred)
    right_term = math.sqrt(max(0, new_rule.var - sigma2))
    significant = left_term > right_term
    if significant:
        new_rule.set_params(significant=True)
    elif epsilon > right_term:
        new_rule.set_params(significant=False)
    else:
        print('Add no-rule')
        
    new_rs = copy.deepcopy(rs)
    new_rs.pop(id_rule)
    new_rs.append(new_rule)
    
    return new_rs


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


def change_rs(rs, bins, xmax, xmin):
    for rg in rs:
        rg_condition = rg.conditions

        var_name = rg_condition.get_param('features_name')
        bmin = rg_condition.get_param('bmin')
        bmax = rg_condition.get_param('bmax')

        if bins is not None:
            i = 0
            for v in var_name:
                if bmin[i] > 0:
                    bmin[i] = bins[v][int(bmin[i] - 1)]
                else:
                    if v == 'x0':
                        bmin[i] = xmin[0]
                    else:
                        bmin[i] = xmin[1]

                if bmax[i] < len(bins[v]):
                    bmax[i] = bins[v][int(bmax[i])]
                else:
                    if v == 'x1':
                        bmax[i] = xmax[0]
                    else:
                        bmax[i] = xmax[1]
                i += 1
                
                
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

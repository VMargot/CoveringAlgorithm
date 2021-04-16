import numpy as np
import pandas as pd
import tqdm
import math
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import rulefit
# 'Install the package rulefit from Christophe Molnar GitHub with the command
# pip install git+git://github.com/christophM/rulefit.git')

import CoveringAlgorithm.CA as CA
import CoveringAlgorithm.covering_tools as ct
from CoveringAlgorithm.functions import extract_rules_rulefit
from ruleskit import RuleSet
from ruleskit.utils.rule_utils import extract_rules_from_tree

import warnings
warnings.filterwarnings("ignore")

seed = 42
np.random.seed(seed)
test_size = 0.3

# RF parameters
max_depth = 3  # number of level by tree
tree_size = 2 ** max_depth  # number of leaves by tree
max_rules = 4000  # total number of rules generated from tree ensembles
nb_estimator = int(np.ceil(max_rules / tree_size))  # Number of tree

# Covering parameters
alpha = 1. / 2 - 1. / 100
gamma = 0.90
lmax = 3
learning_rate = 0.1

nb_simu = 2

info = ['Rules', 'Int', 'mse', 'mse*']
algo = ['DT', 'RF', 'GB', 'AD', 'CA_RF', 'CA_GB', 'CA_AD', 'RuleFit']
cols = [i + ' ' + a for a in algo for i in info]
df = pd.DataFrame(index=range(nb_simu), columns=cols)
dict_count = {'CA_RF': [], 'CA_GB': [], 'CA_AD': []}

for i in tqdm.tqdm(range(nb_simu)):

    # Designing of training data
    nRows = 5000
    nCols = 100

    X = np.random.randint(10, size=(nRows, nCols))
    X = X / 10.

    y_true = 9 * np.exp(-3 * (1 - X[:, 0]) ** 2) * np.exp(-3 * (1 - X[:, 1]) ** 2) * np.exp(-3 * (1 - X[:, 2]) ** 2) \
             - 0.8 * np.exp(-2 * (X[:, 3] - X[:, 4])) + 2 * np.sin(math.pi * X[:, 5]) ** 2 - 2.5 * (X[:, 6] - X[:, 7])

    sigma2 = 1 / 4. * np.var(y_true)  # two-to-one signal-to-noise ratio
    y = y_true + np.random.normal(0, sigma2, nRows)

    features = ['X' + str(j) for j in range(1, nCols + 1)]
    subsample = min(0.5, (100 + 6 * np.sqrt(nRows)) / nRows)

    # Designing of test data
    nRows = 50000
    nCols = 100

    X_test = np.random.randint(10, size=(nRows, nCols))
    X_test = X_test / 10.

    y_true = 9 * np.exp(-3 * (1 - X_test[:, 0]) ** 2) * np.exp(-3 * (1 - X_test[:, 1]) ** 2)\
             * np.exp(-3 * (1 - X_test[:, 2]) ** 2) - 0.8 * np.exp(-2 * (X_test[:, 3] - X_test[:, 4]))\
             + 2 * np.sin(math.pi * X_test[:, 5]) ** 2 - 2.5 * (
                         X_test[:, 6] - X_test[:, 7])

    sigma2_2 = 1 / 4. * np.var(y_true)  # two-to-one signal-to-noise ratio
    y_test = y_true + np.random.normal(0, sigma2_2, nRows)

    deno_y = np.mean(np.abs(y_test - np.mean(y_test))**2)
    deno_ytrue = np.mean(np.abs(y_true - np.mean(y_true))**2)

    # ## Decision Tree
    tree = DecisionTreeRegressor(max_leaf_nodes=10, random_state=seed)
    tree.fit(X, y)

    tree_rules = extract_rules_from_tree(tree, xmins=X.min(axis=0), xmaxs=X.max(axis=0),
                                         features_names=features, get_leaf=True)
    tree_rs = RuleSet(tree_rules)

    # ## Random Forests generation
    regr_rf = RandomForestRegressor(n_estimators=10, random_state=seed)
    regr_rf.fit(X, y)

    rf_rule_list = []
    for tree in regr_rf.estimators_:
        rf_rule_list += extract_rules_from_tree(tree, xmins=X.min(axis=0), xmaxs=X.max(axis=0),
                                                features_names=features, get_leaf=True)
    rf_rs = RuleSet(rf_rule_list)

    # ## Covering Algorithm RandomForest
    ca_rf = CA.CA(alpha=alpha, gamma=gamma,
                  tree_size=tree_size,
                  max_rules=max_rules,
                  generator_func=RandomForestRegressor,
                  lmax=lmax,
                  n_jobs=-1,
                  seed=seed)
    ca_rf.fit(xs=X, y=y, features=features)

    counter = ca_rf.selected_rs.get_variables_count()
    dict_count['CA_RF'].append(counter)

    # ## Covering Algorithm GradientBoosting
    ca_gb = CA.CA(alpha=alpha, gamma=gamma,
                  tree_size=tree_size,
                  max_rules=max_rules,
                  generator_func=GradientBoostingRegressor,
                  lmax=lmax,
                  learning_rate=learning_rate,
                  n_jobs=-1,
                  seed=seed)
    ca_gb.fit(xs=X, y=y, features=features)

    counter = ca_gb.selected_rs.get_variables_count()
    dict_count['CA_GB'].append(counter)

    # ## Covering Algorithm
    ca_ad = CA.CA(alpha=alpha, gamma=gamma,
                  tree_size=tree_size,
                  max_rules=max_rules,
                  generator_func=AdaBoostRegressor,
                  lmax=lmax,
                  subsample=subsample,
                  learning_rate=learning_rate,
                  n_jobs=-1,
                  seed=seed)
    ca_ad.fit(xs=X, y=y, features=features)

    counter = ca_ad.selected_rs.get_variables_count()
    dict_count['CA_AD'].append(counter)

    # ## RuleFit
    rule_fit = rulefit.RuleFit(tree_size=tree_size,
                               max_rules=max_rules,
                               model_type='r',
                               random_state=seed)
    rule_fit.fit(X, y)

    # ### RuleFit rules part
    rules = rule_fit.get_rules()
    rules = rules[rules.coef != 0].sort_values(by="support")
    rules = rules.loc[rules['type'] == 'rule']

    rulefit_rules = extract_rules_rulefit(rules, features, X.min(axis=0), X.max(axis=0))

    # ## Errors calculation
    pred_tree = tree.predict(X_test)
    pred_rf = regr_rf.predict(X_test)
    pred_CA_rf = ca_rf.predict(X_test)
    pred_CA_gb = ca_gb.predict(X_test)
    pred_CA_ad = ca_ad.predict(X_test)
    pred_rulefit = rule_fit.predict(X_test)

    print('Bad prediction for Covering Algorithm RF:',
          sum([x == 0 for x in pred_CA_rf]) / len(y_test))
    print('Bad prediction for Covering Algorithm GB:',
          sum([x == 0 for x in pred_CA_gb]) / len(y_test))
    print('Bad prediction for Covering Algorithm AD:',
          sum([x == 0 for x in pred_CA_ad]) / len(y_test))

    df.iloc[i]['Rules DT'] = len(tree_rules)
    df.iloc[i]['Rules RF'] = len(rf_rule_list)
    df.iloc[i]['Rules CA_RF'] = len(ca_rf.selected_rs)
    df.iloc[i]['Rules CA_GB'] = len(ca_gb.selected_rs)
    df.iloc[i]['Rules CA_AD'] = len(ca_ad.selected_rs)
    df.iloc[i]['Rules RuleFit'] = len(rulefit_rules)

    df.iloc[i]['Int DT'] = ct.interpretability_index(tree_rules)
    df.iloc[i]['Int RF'] = ct.interpretability_index(rf_rule_list)
    df.iloc[i]['Int CA_RF'] = ct.interpretability_index(ca_rf.selected_rs)
    df.iloc[i]['Int CA_GB'] = ct.interpretability_index(ca_gb.selected_rs)
    df.iloc[i]['Int CA_AD'] = ct.interpretability_index(ca_ad.selected_rs)
    df.iloc[i]['Int RuleFit'] = ct.interpretability_index(rulefit_rules)

    df.iloc[i]['mse DT'] = np.mean(np.abs(y_test - pred_tree)**2) / deno_y
    df.iloc[i]['mse* DT'] = np.mean(np.abs(y_true - pred_tree)**2) / deno_ytrue

    df.iloc[i]['mse RF'] = np.mean(np.abs(y_test - pred_rf)**2) / deno_y
    df.iloc[i]['mse* RF'] = np.mean(np.abs(y_true - pred_rf)**2) / deno_ytrue

    df.iloc[i]['mse CA_RF'] = np.mean(np.abs(y_test - pred_CA_rf)**2) / deno_y
    df.iloc[i]['mse* CA_RF'] = np.mean(np.abs(y_true - pred_CA_rf)**2) / deno_ytrue

    df.iloc[i]['mse CA_GB'] = np.mean(np.abs(y_test - pred_CA_gb)**2) / deno_y
    df.iloc[i]['mse* CA_GB'] = np.mean(np.abs(y_true - pred_CA_gb)**2) / deno_ytrue

    df.iloc[i]['mse CA_AD'] = np.mean(np.abs(y_test - pred_CA_ad)**2) / deno_y
    df.iloc[i]['mse* CA_AD'] = np.mean(np.abs(y_true - pred_CA_ad)**2) / deno_ytrue

    df.iloc[i]['mse RuleFit'] = np.mean(np.abs(y_test - pred_rulefit)**2) / deno_y
    df.iloc[i]['mse* RuleFit'] = np.mean(np.abs(y_true - pred_rulefit)**2) / deno_ytrue

print(dict_count)
print(df)

# coding: utf-8
# # Application for the data-dependent covering algorithms on real data
from os.path import dirname, join
import numpy as np
import pandas as pd
import subprocess

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import rulefit
# 'Install the package rulefit from Christophe Molnar GitHub with the command
# pip install git+git://github.com/christophM/rulefit.git')

import CoveringAlgorithm.CA as CA
import CoveringAlgorithm.covering_tools as ct
from CoveringAlgorithm.functions import make_rs_from_r, extract_rules_rulefit
from Data.load_data import load_data, target_dict
from ruleskit.ruleset import RuleSet
from ruleskit.utils.rule_utils import extract_rules_from_tree


import warnings
warnings.filterwarnings("ignore")

racine_path = dirname(__file__)

pathx = join(racine_path, 'X.csv')
pathx_test = join(racine_path, 'X_test.csv')
pathy = join(racine_path, 'Y.csv')
pathr = join(racine_path, 'main.r')
r_script = '/usr/bin/Rscript'

if __name__ == '__main__':
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
    res_dict = {}
    #  Data parameters
    seed = 42
    for data_name in ['prostate', 'diabetes', 'ozone', 'machine', 'mpg',
                      'boston', 'student_por', 'abalone']:
        print('')
        print('===== ', data_name.upper(), ' =====')

        # ## Data Generation
        dataset = load_data(data_name)
        target = target_dict[data_name]
        y = dataset[target].astype('float')
        X = dataset.drop(target, axis=1)
        features = X.describe().columns
        X = X[features]

        res_dict['DT'] = []
        res_dict['RF'] = []
        res_dict['CA_RF'] = []
        res_dict['CA_GB'] = []
        res_dict['CA_SGB'] = []
        res_dict['RuleFit'] = []
        res_dict['Sirus'] = []
        res_dict['NH'] = []
        p0 = '0'
        for i in range(nb_simu):
            # ### Splitting data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=seed)
            if test_size == 0.0:
                X_test = X_train
                y_test = y_train

            X_train.to_csv(pathx, index=False)
            y_train.to_csv(pathy, index=False, header=False)
            X_test.to_csv(pathx_test, index=False)

            y_train = y_train.values
            y_test = y_test.values
            X_train = X_train.values  # To get only numerical variables
            X_test = X_test.values

            with open('output_rfile.txt', 'w') as f:
                subprocess.call([r_script, "--no-save", "--no-restore",
                                 "--verbose", "--vanilla", pathr,
                                 pathx, pathy, pathx_test, p0, str(seed)],
                                stdout=f, stderr=subprocess.STDOUT)

            if p0 == '0':
                f = open("p0.txt", "r")
                p0 = f.readline()
                p0 = p0.replace('\n', '')
                f.close()

            pred_sirus = pd.read_csv(join(racine_path, 'sirus_pred.csv'))['x'].values
            pred_nh = pd.read_csv(join(racine_path, 'nh_pred.csv'))['x'].values
            rules_sirus = pd.read_csv(join(racine_path, 'sirus_rules.csv'))
            rules_nh = pd.read_csv(join(racine_path, 'nh_rules.csv'))

            sirus_rs = make_rs_from_r(rules_sirus, features.to_list(), X_train.min(axis=0),
                                      X_train.max(axis=0))
            nh_rs = make_rs_from_r(rules_nh, features.to_list(), X_train.min(axis=0),
                                   X_train.max(axis=0))
            nh_rs = RuleSet(nh_rs[:-1])

            # Normalization of the error
            deno_mse = np.mean((y_test - np.mean(y_test)) ** 2)

            subsample = min(0.5, (100 + 6 * np.sqrt(len(y_train))) / len(y_train))

            # ## Decision Tree
            tree = DecisionTreeRegressor(max_leaf_nodes=10, random_state=seed)
            tree.fit(X_train, y_train)

            tree_rules = extract_rules_from_tree(tree, xmins=X_train.min(axis=0), xmaxs=X_train.max(axis=0),
                                                 features_names=features, get_leaf=True)
            tree_rs = RuleSet(tree_rules)

            # Random Forests generation
            regr_rf = RandomForestRegressor(n_estimators=1000, random_state=seed)
            regr_rf.fit(X_train, y_train)

            rf_rule_list = []
            for tree in regr_rf.estimators_:
                rf_rule_list += extract_rules_from_tree(tree, xmins=X_train.min(axis=0), xmaxs=X_train.max(axis=0),
                                                        features_names=features, get_leaf=True)
            rf_rs = RuleSet(rf_rule_list)

            seed += 1
            # Covering Algorithm RandomForest
            ca_rf = CA.CA(alpha=alpha, gamma=gamma,
                          tree_size=tree_size,
                          max_rules=max_rules,
                          generator_func=RandomForestRegressor,
                          lmax=lmax,
                          seed=seed)
            ca_rf.fit(xs=X_train, y=y_train, features=features)

            seed += 1
            # ## Covering Algorithm GradientBoosting
            ca_gb = CA.CA(alpha=alpha, gamma=gamma,
                          tree_size=tree_size,
                          max_rules=max_rules,
                          generator_func=GradientBoostingRegressor,
                          lmax=lmax,
                          learning_rate=learning_rate,
                          seed=seed)
            ca_gb.fit(xs=X_train, y=y_train, features=features)

            seed += 1
            # ## Covering Algorithm StochasicGradientBoosting
            n = X_train.shape[0]
            subsample = min(0.5, (100 + 6 * np.sqrt(n)) / n)
            ca_sgb = CA.CA(alpha=alpha, gamma=gamma,
                           tree_size=tree_size,
                           max_rules=max_rules,
                           generator_func=GradientBoostingRegressor,
                           lmax=lmax,
                           subsample=subsample,
                           learning_rate=learning_rate,
                           seed=seed)
            ca_sgb.fit(xs=X_train, y=y_train, features=features)

            seed += 1
            # ## RuleFit
            rule_fit = rulefit.RuleFit(tree_size=tree_size,
                                       max_rules=max_rules,
                                       model_type='r',
                                       random_state=seed)
            rule_fit.fit(X_train, y_train)

            # ### RuleFit rules part
            model = rule_fit.get_rules()
            model = model[model.coef != 0].sort_values(by="support")
            rules = model.loc[model['type'] == 'rule']
            # ### RuleFit linear part
            lin = model.loc[model['type'] == 'linear']

            rulefit_rs = extract_rules_rulefit(rules, features, X_train.min(axis=0),
                                               X_train.max(axis=0))

            #  ## Errors calculation
            pred_tree = tree.predict(X_test)
            pred_rf = regr_rf.predict(X_test)
            pred_CA_rf = ca_rf.predict(X_test)
            pred_CA_gb = ca_gb.predict(X_test)
            pred_CA_sgb = ca_sgb.predict(X_test)
            pred_rulefit = rule_fit.predict(X_test)

            mse_tree = np.mean((y_test - pred_tree) ** 2) / deno_mse
            mse_rf = np.mean((y_test - pred_rf) ** 2) / deno_mse
            mse_CA_rf = np.mean((y_test - pred_CA_rf) ** 2) / deno_mse
            mse_CA_gb = np.mean((y_test - pred_CA_gb) ** 2) / deno_mse
            mse_CA_sgb = np.mean((y_test - pred_CA_sgb) ** 2) / deno_mse
            mse_rulefit = np.mean((y_test - pred_rulefit) ** 2) / deno_mse
            mse_sirus = np.mean((y_test - pred_sirus) ** 2) / deno_mse
            mse_nh = np.mean((y_test - pred_nh) ** 2) / deno_mse

            cov_tree = tree_rs.calc_coverage_rate(X_train)
            cov_rf = rf_rs.calc_coverage_rate(X_train)
            cov_CA_rf = ca_rf.selected_rs.calc_coverage_rate()
            cov_CA_gb = ca_gb.selected_rs.calc_coverage_rate()
            cov_CA_sgb = ca_sgb.selected_rs.calc_coverage_rate()
            cov_rulefit = rulefit_rs.calc_coverage_rate(X_train)

            cov_sirus = sirus_rs.calc_coverage_rate(X_train)
            cov_nh = nh_rs.calc_coverage_rate(X_train)

            seed += 1

            if i == 0:
                res_dict['DT'] = [[len(tree_rules), cov_tree, ct.interpretability_index(tree_rules),
                                   r2_score(y_test, pred_tree), mse_tree]]
                res_dict['RF'] = [[len(rf_rule_list), cov_rf, ct.interpretability_index(rf_rule_list),
                                   r2_score(y_test, pred_rf), mse_rf]]
                res_dict['CA_RF'] = [[len(ca_rf.selected_rs), cov_CA_rf,
                                      ct.interpretability_index(ca_rf.selected_rs), r2_score(y_test, pred_CA_rf),
                                      mse_CA_rf]]
                res_dict['CA_GB'] = [[len(ca_gb.selected_rs), cov_CA_gb,
                                      ct.interpretability_index(ca_gb.selected_rs), r2_score(y_test, pred_CA_gb),
                                      mse_CA_gb]]
                res_dict['CA_SGB'] = [[len(ca_sgb.selected_rs), cov_CA_sgb,
                                       ct.interpretability_index(ca_sgb.selected_rs), r2_score(y_test, pred_CA_sgb),
                                       mse_CA_sgb]]
                res_dict['RuleFit'] = [[len(rulefit_rs), cov_rulefit,
                                        ct.interpretability_index(rulefit_rs), r2_score(y_test, pred_rulefit),
                                        mse_rulefit]]
                res_dict['Sirus'] = [[len(sirus_rs), cov_sirus, ct.interpretability_index(sirus_rs),
                                      r2_score(y_test, pred_sirus), mse_sirus]]
                res_dict['NH'] = [[len(nh_rs), cov_nh, ct.interpretability_index(nh_rs), r2_score(y_test, pred_nh),
                                   mse_nh]]

            else:
                res_dict['DT'] = np.append(res_dict['DT'], [[len(tree_rules), cov_tree,
                                                             ct.interpretability_index(tree_rules),
                                                             r2_score(y_test, pred_tree),
                                                             mse_tree]], axis=0)
                res_dict['RF'] = np.append(res_dict['RF'], [[len(rf_rule_list), cov_rf,
                                                             ct.interpretability_index(rf_rule_list),
                                                             r2_score(y_test, pred_rf),
                                                             mse_rf]], axis=0)
                res_dict['CA_RF'] = np.append(res_dict['CA_RF'], [[len(ca_rf.selected_rs),
                                                                   cov_CA_rf,
                                                                   ct.interpretability_index(ca_rf.selected_rs),
                                                                   r2_score(y_test, pred_CA_rf),
                                                                   mse_CA_rf]], axis=0)
                res_dict['CA_GB'] = np.append(res_dict['CA_GB'], [[len(ca_gb.selected_rs),
                                                                   cov_CA_gb,
                                                                   ct.interpretability_index(ca_gb.selected_rs),
                                                                   r2_score(y_test, pred_CA_gb),
                                                                   mse_CA_gb]], axis=0)
                res_dict['CA_SGB'] = np.append(res_dict['CA_SGB'], [[len(ca_sgb.selected_rs),
                                                                     cov_CA_sgb,
                                                                     ct.interpretability_index(ca_sgb.selected_rs),
                                                                     r2_score(y_test, pred_CA_sgb),
                                                                     mse_CA_sgb]], axis=0)
                res_dict['RuleFit'] = np.append(res_dict['RuleFit'], [[len(rulefit_rs),
                                                                       cov_rulefit,
                                                                       ct.interpretability_index(rulefit_rs),
                                                                       r2_score(y_test,
                                                                                pred_rulefit),
                                                                       mse_rulefit]], axis=0)
                res_dict['Sirus'] = np.append(res_dict['Sirus'], [[len(sirus_rs), cov_sirus,
                                                                   ct.interpretability_index(sirus_rs),
                                                                   r2_score(y_test, pred_sirus),
                                                                   mse_sirus]], axis=0)
                res_dict['NH'] = np.append(res_dict['NH'], [[len(nh_rs), cov_nh, ct.interpretability_index(nh_rs),
                                                             r2_score(y_test, pred_nh),
                                                             mse_nh]], axis=0)

        # ## Results.
        print('')
        print('Nb Rules')
        print('----------------------')
        print('Decision tree nb rules:', np.mean(res_dict['DT'][:, 0]))
        print('Random Forest nb rules:', np.mean(res_dict['RF'][:, 0]))
        print('Covering Algorithm RF nb rules:', np.mean(res_dict['CA_RF'][:, 0]))
        print('Covering Algorithm GB nb rules:', np.mean(res_dict['CA_GB'][:, 0]))
        print('Covering Algorithm SGB nb rules:', np.mean(res_dict['CA_SGB'][:, 0]))
        print('RuleFit nb rules:', np.mean(res_dict['RuleFit'][:, 0]))
        print('SIRUS nb rules:', np.mean(res_dict['Sirus'][:, 0]))
        print('NodeHarvest nb rules:', np.mean(res_dict['NH'][:, 0]))

        print('')
        print('Coverage')
        print('----------------------')
        print('Decision tree coverage:', np.mean(res_dict['DT'][:, 1]))
        print('Random Forest coverage:', np.mean(res_dict['RF'][:, 1]))
        print('Covering Algorithm RF coverage:', np.mean(res_dict['CA_RF'][:, 1]))
        print('Covering Algorithm GB coverage:', np.mean(res_dict['CA_GB'][:, 1]))
        print('Covering Algorithm SGB coverage:', np.mean(res_dict['CA_SGB'][:, 1]))
        print('RuleFit coverage:', np.mean(res_dict['RuleFit'][:, 1]))
        print('SIRUS coverage:', np.mean(res_dict['Sirus'][:, 1]))
        print('NodeHarvest coverage:', np.mean(res_dict['NH'][:, 1]))

        print('')
        print('Interpretability score')
        print('----------------------')
        print('Decision tree interpretability score:', np.mean(res_dict['DT'][:, 2]))
        print('Random Forest interpretability score:', np.mean(res_dict['RF'][:, 2]))
        print('Covering Algorithm RF interpretability score:', np.mean(res_dict['CA_RF'][:, 2]))
        print('Covering Algorithm GB interpretability score:', np.mean(res_dict['CA_GB'][:, 2]))
        print('Covering Algorithm SGB interpretability score:', np.mean(res_dict['CA_SGB'][:, 2]))
        print('RuleFit interpretability score:', np.mean(res_dict['RuleFit'][:, 2]))
        print('SIRUS interpretability score:', np.mean(res_dict['Sirus'][:, 2]))
        print('NodeHarvest interpretability score:', np.mean(res_dict['NH'][:, 2]))

        print('')
        print('R2 score')  # Percentage of the unexplained variance
        print('--------')
        print('Decision tree R2 score:', 1.0 - np.mean(res_dict['DT'][:, 3]))
        print('Random Forest R2 score:', 1.0 - np.mean(res_dict['RF'][:, 3]))
        print('Covering Algorithm RF R2 score:', 1.0 - np.mean(res_dict['CA_RF'][:, 3]))
        print('Covering Algorithm GB R2 score:', 1.0 - np.mean(res_dict['CA_GB'][:, 3]))
        print('Covering Algorithm SGB R2 score:', 1.0 - np.mean(res_dict['CA_SGB'][:, 3]))
        print('RuleFit R2 score:', 1.0 - np.mean(res_dict['RuleFit'][:, 3]))
        print('SIRUS R2 score:', 1.0 - np.mean(res_dict['Sirus'][:, 3]))
        print('NodeHarvest R2 score:', 1.0 - np.mean(res_dict['NH'][:, 3]))

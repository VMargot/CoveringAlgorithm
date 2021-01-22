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
from CoveringAlgorithm.ruleset import RuleSet


import warnings
warnings.filterwarnings("ignore")

target_dict = {'student_mat': 'G3',
               'student_por': 'G3',
               'student_mat_easy': 'G3',
               'student_por_easy': 'G3',
               'boston': 'MEDV',
               'bike_hour': 'cnt',
               'bike_day': 'cnt',
               'mpg': 'mpg',
               'machine': 'ERP',
               'abalone': 'Rings',
               'prostate': 'lpsa',
               'ozone': 'ozone',
               'diabetes': 'Y'}

racine_path = dirname(__file__)

pathx = join(racine_path, 'X.csv')
pathx_test = join(racine_path, 'X_test.csv')
pathy = join(racine_path, 'Y.csv')
pathr = join(racine_path, 'main.r')
r_script = '/usr/bin/Rscript'


def load_data(name: str):
    """
    Parameters
    ----------
    name: a chosen data set

    Returns
    -------
    data: a pandas DataFrame
    """
    if 'student' in name:
        if 'student_por' in name:
            data = pd.read_csv(join(racine_path, 'Data/Student/student-por.csv'),
                               sep=';')
        elif 'student_mat' in name:
            data = pd.read_csv(join(racine_path, 'Data/Student/student-mat.csv'),
                               sep=';')
        else:
            raise ValueError('Not tested dataset')
        # Covering Algorithm allow only numerical features.
        # We can only convert binary qualitative features.
        data['sex'] = [1 if x == 'F' else 0 for x in data['sex'].values]
        data['Pstatus'] = [1 if x == 'A' else 0 for x in data['Pstatus'].values]
        data['famsize'] = [1 if x == 'GT3' else 0 for x in data['famsize'].values]
        data['address'] = [1 if x == 'U' else 0 for x in data['address'].values]
        data['school'] = [1 if x == 'GP' else 0 for x in data['school'].values]
        data = data.replace('yes', 1)
        data = data.replace('no', 0)

        if 'easy' not in data_name:
            # For an harder exercise drop G1 and G2
            data = data.drop(['G1', 'G2'], axis=1)

    elif name == 'bike_hour':
        data = pd.read_csv(join(racine_path, 'Data/BikeSharing/hour.csv'), index_col=0)
        data = data.set_index('dteday')
    elif name == 'bike_day':
        data = pd.read_csv(join(racine_path, 'Data/BikeSharing/day.csv'), index_col=0)
        data = data.set_index('dteday')
    elif name == 'mpg':
        data = pd.read_csv(join(racine_path, 'Data/MPG/mpg.csv'))
    elif name == 'machine':
        data = pd.read_csv(join(racine_path, 'Data/Machine/machine.csv'))
    elif name == 'abalone':
        data = pd.read_csv(join(racine_path, 'Data/Abalone/abalone.csv'))
    elif name == 'ozone':
        data = pd.read_csv(join(racine_path, 'Data/Ozone/ozone.csv'))
    elif name == 'prostate':
        data = pd.read_csv(join(racine_path, 'Data/Prostate/prostate.csv'), index_col=0)
    elif name == 'diabetes':
        data = pd.read_csv(join(racine_path, 'Data/Diabetes/diabetes.csv'), index_col=0)
    elif name == 'boston':
        from sklearn.datasets import load_boston
        boston_dataset = load_boston()
        data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
        data['MEDV'] = boston_dataset.target
    else:
        raise ValueError('Not tested dataset')

    return data.dropna()


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

    nb_simu = 10
    res_dict = {}
    #  Data parameters
    seed = 42
    for data_name in ['prostate', 'diabetes', 'prostate', 'ozone',
                      'machine', 'mpg', 'boston', 'student_por', 'abalone']:
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

            sirus_rs = ct.make_rs_from_r(rules_sirus, features.to_list(), X_train.min(axis=0),
                                         X_train.max(axis=0))
            nh_rs = ct.make_rs_from_r(rules_nh, features.to_list(), X_train.min(axis=0),
                                      X_train.max(axis=0))
            nh_rs = RuleSet(nh_rs[:-1])

            # Normalization of the error
            deno_mse = np.mean((y_test - np.mean(y_test)) ** 2)

            subsample = min(0.5, (100 + 6 * np.sqrt(len(y_train))) / len(y_train))

            # ## Decision Tree
            tree = DecisionTreeRegressor(max_leaf_nodes=10, random_state=seed)
            tree.fit(X_train, y_train)

            tree_rules = ct.extract_rules_from_tree(tree, features, X_train.min(axis=0),
                                                    X_train.max(axis=0), get_leaf=True)
            tree_rs = RuleSet(tree_rules)

            # ## Random Forests generation
            regr_rf = RandomForestRegressor(n_estimators=1000, random_state=seed)
            regr_rf.fit(X_train, y_train)

            rf_rule_list = []
            for tree in regr_rf.estimators_:
                rf_rule_list += ct.extract_rules_from_tree(tree, features, X_train.min(axis=0),
                                                           X_train.max(axis=0), get_leaf=True)
            rf_rs = RuleSet(rf_rule_list)

            seed += 1
            # Covering Algorithm RandomForest
            ca_rf = CA.CA(alpha=alpha, gamma=gamma,
                          tree_size=tree_size,
                          max_rules=max_rules,
                          generator_func=RandomForestRegressor,
                          lmax=lmax,
                          seed=seed)
            ca_rf.fit(X=X_train, y=y_train, features=features)

            seed += 1
            # ## Covering Algorithm GradientBoosting
            ca_gb = CA.CA(alpha=alpha, gamma=gamma,
                          tree_size=tree_size,
                          max_rules=max_rules,
                          generator_func=GradientBoostingRegressor,
                          lmax=lmax,
                          learning_rate=learning_rate,
                          seed=seed)
            ca_gb.fit(X=X_train, y=y_train, features=features)

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
            ca_sgb.fit(X=X_train, y=y_train, features=features)

            seed += 1
            # ## RuleFit
            rule_fit = rulefit.RuleFit(tree_size=tree_size,
                                       max_rules=max_rules,
                                       model_type='r',
                                       random_state=seed)
            rule_fit.fit(X_train, y_train)

            # ### RuleFit rules part
            rules = rule_fit.get_rules()
            rules = rules[rules.coef != 0].sort_values(by="support")
            rules = rules.loc[rules['type'] == 'rule']

            # ### RuleFit linear part
            lin = rule_fit.get_rules()
            lin = lin[lin.coef != 0].sort_values(by="support")
            lin = lin.loc[lin['type'] == 'linear']

            rulefit_rules = ct.extract_rules_rulefit(rules, features, X_train.min(axis=0),
                                                     X_train.max(axis=0))
            rulefit_rs = RuleSet(rulefit_rules)

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

            cov_tree = tree_rs.calc_coverage(X_train)
            cov_rf = rf_rs.calc_coverage(X_train)
            cov_CA_rf = ca_rf.selected_rs.calc_coverage()
            cov_CA_gb = ca_gb.selected_rs.calc_coverage()
            cov_CA_sgb = ca_sgb.selected_rs.calc_coverage()
            cov_rulefit = rulefit_rs.calc_coverage(X_train)
            cov_sirus = sirus_rs.calc_coverage(X_train)
            cov_nh = nh_rs.calc_coverage(X_train)

            seed += 1

            if i == 0:
                res_dict['DT'] = [[len(tree_rules), cov_tree, ct.inter(tree_rules),
                                   r2_score(y_test, pred_tree), mse_tree]]
                res_dict['RF'] = [[len(rf_rule_list), cov_rf, ct.inter(rf_rule_list),
                                   r2_score(y_test, pred_rf), mse_rf]]
                res_dict['CA_RF'] = [[len(ca_rf.selected_rs), cov_CA_rf,
                                      ct.inter(ca_rf.selected_rs),  r2_score(y_test, pred_CA_rf),
                                      mse_CA_rf]]
                res_dict['CA_GB'] = [[len(ca_gb.selected_rs), cov_CA_gb,
                                      ct.inter(ca_gb.selected_rs), r2_score(y_test, pred_CA_gb),
                                      mse_CA_gb]]
                res_dict['CA_SGB'] = [[len(ca_sgb.selected_rs), cov_CA_sgb,
                                      ct.inter(ca_sgb.selected_rs), r2_score(y_test, pred_CA_sgb),
                                      mse_CA_sgb]]
                res_dict['RuleFit'] = [[len(rulefit_rules), cov_rulefit,
                                        ct.inter(rulefit_rules), r2_score(y_test, pred_rulefit),
                                        mse_rulefit]]
                res_dict['Sirus'] = [[len(sirus_rs), cov_sirus, ct.inter(sirus_rs),
                                      r2_score(y_test, pred_sirus), mse_sirus]]
                res_dict['NH'] = [[len(nh_rs), cov_nh, ct.inter(nh_rs), r2_score(y_test, pred_nh),
                                   mse_nh]]

            else:
                res_dict['DT'] = np.append(res_dict['DT'], [[len(tree_rules), cov_tree,
                                                             ct.inter(tree_rules),
                                                             r2_score(y_test, pred_tree),
                                                             mse_tree]], axis=0)
                res_dict['RF'] = np.append(res_dict['RF'], [[len(rf_rule_list), cov_rf,
                                                             ct.inter(rf_rule_list),
                                                             r2_score(y_test, pred_rf),
                                                             mse_rf]], axis=0)
                res_dict['CA_RF'] = np.append(res_dict['CA_RF'], [[len(ca_rf.selected_rs),
                                                                   cov_CA_rf,
                                                                   ct.inter(ca_rf.selected_rs),
                                                                   r2_score(y_test, pred_CA_rf),
                                                                   mse_CA_rf]], axis=0)
                res_dict['CA_GB'] = np.append(res_dict['CA_GB'], [[len(ca_gb.selected_rs),
                                                                   cov_CA_gb,
                                                                   ct.inter(ca_gb.selected_rs),
                                                                   r2_score(y_test, pred_CA_gb),
                                                                   mse_CA_gb]], axis=0)
                res_dict['CA_SGB'] = np.append(res_dict['CA_SGB'], [[len(ca_sgb.selected_rs),
                                                                     cov_CA_sgb,
                                                                     ct.inter(ca_sgb.selected_rs),
                                                                     r2_score(y_test, pred_CA_sgb),
                                                                     mse_CA_sgb]], axis=0)
                res_dict['RuleFit'] = np.append(res_dict['RuleFit'], [[len(rulefit_rules),
                                                                       cov_rulefit,
                                                                       ct.inter(rulefit_rules),
                                                                       r2_score(y_test,
                                                                                pred_rulefit),
                                                                       mse_rulefit]], axis=0)
                res_dict['Sirus'] = np.append(res_dict['Sirus'], [[len(sirus_rs), cov_sirus,
                                                                   ct.inter(sirus_rs),
                                                                   r2_score(y_test, pred_sirus),
                                                                   mse_sirus]], axis=0)
                res_dict['NH'] = np.append(res_dict['NH'], [[len(nh_rs), cov_nh, ct.inter(nh_rs),
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

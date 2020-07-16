import tqdm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from CoveringAlgorithm import covering_tools as ct
from CoveringAlgorithm.ruleset import RuleSet


class CA(object):

    def __init__(self, **kwargs):
        self.rule_list = []
        self.lmax = 3
        self.alpha = 1/2 - 1/100
        self.gamma = 0.95
        self.selected_rs = RuleSet([])
        self.rule_generator = RandomForestRegressor(**kwargs)

    def fit(self, x, y, features=None):
        xmin = x.min(axis=0)
        xmax = x.max(axis=0)
        if features is None:
            features = ['feature_' + str(x) for x in range(0, x.shape[1])]

        self.rule_generator.fit(x, y)
        self.extract_rules(xmin, xmax, features)
        self.eval_rules(x, y)
        self.select_rules(y)

    def extract_rules(self, xmin, xmax, features):
        for tree in self.rule_generator.estimators_:
            self.rule_list += ct.extract_rules_from_tree(tree, features,
                                                         xmin, xmax)

    def eval_rules(self, x, y):
        [rule.calc_stats(y=y, x=x, cov_min=0.0, cov_max=1.0)
         for rule in tqdm.tqdm(self.rule_list)]

    def select_rules(self, y):
        sub_rulelist = list(filter(lambda rule: rule.length <= self.lmax, self.rule_list))
        sigma = self.get_sigma(len(y))
        self.selected_rs = ct.find_covering(sub_rulelist, y, sigma,
                                            self.alpha, self.gamma)

    def get_sigma(self, n_train):
        sigma = np.nanmin([r.var if r.cov > n_train ** (-self.alpha) else np.nan
                           for r in self.rule_list])
        return sigma

    def predict(self, ytrain, x):
        prediction_vector = ct.calc_pred(self.selected_rs, ytrain, x)
        return prediction_vector

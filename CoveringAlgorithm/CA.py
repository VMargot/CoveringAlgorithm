import tqdm
from typing import List, Callable
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,\
    GradientBoostingClassifier, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from CoveringAlgorithm import covering_tools as ct
from CoveringAlgorithm.ruleset import RuleSet


def eval_rules(rule, y, x):
    rule.calc_stats(x=x, y=y)
    return rule

class CA(object):

    def __init__(self, alpha: float = 1/2 - 1/100, gamma: float = 0.95,
                 lmax: int = 3, tree_size: int = 4, max_rules: int = 2000,
                 learning_rate: float = 0.01, n_jobs: int = None, seed: int = None,
                 mode: str = 'r', generator_func: Callable = None):
        """
        Parameters
        ----------
        alpha
        gamma
        lmax
        tree_size
        max_rules
        learning_rate
        n_jobs
        seed
        generator_func
        """

        self.l_max = lmax
        self.alpha = alpha
        self.gamma = gamma
        self.tree_size = tree_size
        self.max_rules = max_rules
        self.n_jobs = n_jobs
        self.seed = seed
        self.learning_rate = learning_rate
        self.mode = mode
        self.generator = generator_func
        self.rule_generator = None
        self.rule_list = []
        self.selected_rs = RuleSet([])

    def fit(self, x: np.ndarray, y: np.ndarray, features: List[str] = None):
        xmin = x.min(axis=0)
        xmax = x.max(axis=0)
        if features is None:
            features = ['feature_' + str(x) for x in range(0, x.shape[1])]

        n = x.shape[0]
        subsample = min(0.5, (100 + 6 * np.sqrt(n)) / n)
        nb_estimator = int(np.ceil(self.max_rules/self.tree_size))

        self.set_rule_generator(nb_estimator, subsample, self.mode)
        self.rule_generator.fit(x, y)
        self.extract_rules(xmin, xmax, features)
        self.eval_rules(x, y)
        self.select_rules(y)

    def extract_rules(self, x_min: List[float], x_max: List[float], features: List[str]):
        if type(self.rule_generator) in [GradientBoostingRegressor, GradientBoostingClassifier]:
            tree_list = [x[0] for x in self.rule_generator.estimators_]
        else:
            tree_list = self.rule_generator.estimators_
        for tree in tree_list:
            self.rule_list += ct.extract_rules_from_tree(tree, features, x_min, x_max)

    def eval_rules(self, x: np.ndarray, y: np.ndarray):
        self.rule_list = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(eval_rules)(rule, y, x) for rule in tqdm.tqdm(self.rule_list))

    def select_rules(self, y: np.ndarray):
        sub_rulelist = list(filter(lambda rule: rule.length <= self.l_max, self.rule_list))
        sigma = self.get_sigma(len(y))
        self.selected_rs = ct.find_covering(sub_rulelist, y, sigma,
                                            self.alpha, self.gamma)

    def set_rule_generator(self, nb_estimator, subsample, mode):
        if self.generator is None:
            if mode.lower() in ['regression', 'reg', 'r']:
                self.rule_generator = GradientBoostingRegressor(n_estimators=nb_estimator,
                                                                max_leaf_nodes=self.tree_size,
                                                                learning_rate=self.learning_rate,
                                                                subsample=subsample,
                                                                random_state=self.seed,
                                                                max_depth=100)
            elif mode.lower() in ['classification', 'classif', 'c']:
                self.rule_generator = GradientBoostingClassifier(n_estimators=nb_estimator,
                                                                 max_leaf_nodes=self.tree_size,
                                                                 learning_rate=self.learning_rate,
                                                                 subsample=subsample,
                                                                 random_state=self.seed,
                                                                 max_depth=100)
            else:
                raise ValueError('Covering Algorithm only works for regression or classification.')

        elif self.generator in [RandomForestClassifier, RandomForestRegressor]:
            self.rule_generator = self.generator(n_estimators=nb_estimator,
                                                 max_leaf_nodes=self.tree_size,
                                                 random_state=self.seed,
                                                 max_depth=100)
        elif self.generator in [GradientBoostingRegressor, GradientBoostingClassifier]:
            self.rule_generator = self.generator(n_estimators=nb_estimator,
                                                 max_leaf_nodes=self.tree_size,
                                                 learning_rate=self.learning_rate,
                                                 subsample=subsample,
                                                 random_state=self.seed,
                                                 max_depth=100)
        elif self.generator in [AdaBoostRegressor, AdaBoostClassifier]:
            self.rule_generator = self.generator(n_estimators=nb_estimator,
                                                 learning_rate=self.learning_rate,
                                                 random_state=self.seed)
        else:
            raise ValueError("Covering Algorithm only works with "
                             "RandomForest, GradientBoosting and AdBoost!")

    def get_sigma(self, n_train: int):
        sigma = np.nanmin([r.var if r.cov > n_train ** (-self.alpha) else np.nan
                           for r in self.rule_list])
        return sigma

    def predict(self, ytrain: np.ndarray, x: np.ndarray):
        prediction_vector = ct.calc_pred(self.selected_rs, ytrain, x)
        return prediction_vector

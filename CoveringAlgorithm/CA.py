
from typing import List, Callable
import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.validation import check_X_y, check_array
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,\
    GradientBoostingClassifier, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from CoveringAlgorithm import covering_tools as ct
from CoveringAlgorithm import functions as f
from CoveringAlgorithm.ruleset import RuleSet
from CoveringAlgorithm.rule import Rule


def eval_rules(rule: Rule, y: np.ndarray, X: np.ndarray):
    """
    Parameters
    ----------
    rule: rule to evaluate
    y: variable of interest
    X: features matrix

    Returns
    -------
    rule: rule evaluated on (X, y)
    """
    rule.calc_stats(x=X, y=y)
    return rule


class CA:
    """
    Covering Algorithm class
    """
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
        self.rules_generator = None
        self.features = []
        self.rules_list = []
        self.selected_rs = RuleSet([])
        self.y = None

    def _validate_X_predict(self, X: np.ndarray, check_input: bool = True):
        """Validate X whenever one tries to predict"""
        if check_input:
            X = check_array(X, accept_sparse=True, force_all_finite='allow-nan')

        n_features = X.shape[1]
        if len(self.features) != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (len(self.features), n_features))

        return X

    def fit(self, X: np.ndarray, y: np.ndarray, features: List[str] = None):
        """
        Build a covering algorithm from the set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        features : array-like of shape (n_features,), default=None
                   Name of the features with the same order
        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, ensure_min_samples=10, accept_sparse=True,
                         force_all_finite='allow-nan', y_numeric=True)
        self.y = y

        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        if features is None:
            self.features = ['feature_' + str(col) for col in range(0, X.shape[1])]
        else:
            self.features = features
        n = X.shape[0]
        subsample = min(0.5, (100 + 6 * np.sqrt(n)) / n)
        nb_estimator = int(np.ceil(self.max_rules/self.tree_size))

        self.set_rule_generator(nb_estimator, subsample, self.mode)
        self.rules_generator.fit(X, y)
        self.extract_rules(x_min, x_max)
        self.eval_rules(X, y)
        self.select_rules(y)

        return self

    def extract_rules(self, x_min: List[float], x_max: List[float]):
        if type(self.rules_generator) in [GradientBoostingRegressor, GradientBoostingClassifier]:
            tree_list = [t[0] for t in self.rules_generator.estimators_]
        else:
            tree_list = self.rules_generator.estimators_
        for tree in tree_list:
            self.rules_list += ct.extract_rules_from_tree(tree, self.features, x_min, x_max)

    def eval_rules(self, X: np.ndarray, y: np.ndarray):
        self.rules_list = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(eval_rules)(rule, y, X) for rule in self.rules_list)

    def select_rules(self, y: np.ndarray):
        sub_rulelist = list(filter(lambda rule: rule.length <= self.l_max, self.rules_list))
        sigma = self.get_sigma(len(y))
        self.selected_rs = ct.find_covering(sub_rulelist, y, sigma,
                                            self.alpha, self.gamma)

    def set_rule_generator(self, nb_estimator, subsample, mode):
        if self.generator is None:
            if mode.lower() in ['regression', 'reg', 'r']:
                self.rules_generator = GradientBoostingRegressor(n_estimators=nb_estimator,
                                                                 max_leaf_nodes=self.tree_size,
                                                                 learning_rate=self.learning_rate,
                                                                 subsample=subsample,
                                                                 random_state=self.seed,
                                                                 max_depth=100)
            elif mode.lower() in ['classification', 'classif', 'c']:
                self.rules_generator = GradientBoostingClassifier(n_estimators=nb_estimator,
                                                                  max_leaf_nodes=self.tree_size,
                                                                  learning_rate=self.learning_rate,
                                                                  subsample=subsample,
                                                                  random_state=self.seed,
                                                                  max_depth=100)
            else:
                raise ValueError('Covering Algorithm only works for regression or classification.')

        elif self.generator in [RandomForestClassifier, RandomForestRegressor]:
            self.rules_generator = self.generator(n_estimators=nb_estimator,
                                                  max_leaf_nodes=self.tree_size,
                                                  random_state=self.seed,
                                                  max_depth=100)
        elif self.generator in [GradientBoostingRegressor, GradientBoostingClassifier]:
            self.rules_generator = self.generator(n_estimators=nb_estimator,
                                                  max_leaf_nodes=self.tree_size,
                                                  learning_rate=self.learning_rate,
                                                  subsample=subsample,
                                                  random_state=self.seed,
                                                  max_depth=100)
        elif self.generator in [AdaBoostRegressor, AdaBoostClassifier]:
            self.rules_generator = self.generator(n_estimators=nb_estimator,
                                                  learning_rate=self.learning_rate,
                                                  random_state=self.seed)
        else:
            raise ValueError("Covering Algorithm only works with "
                             "RandomForest, GradientBoosting and AdBoost!")

    def get_sigma(self, n_train: int):
        sigma = np.nanmin([r.var if r.cov > n_train ** (-self.alpha) else np.nan
                           for r in self.rules_list])
        return sigma

    def predict(self, X: np.ndarray, y: np.ndarray = None):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values real numbers in regression used to calculate the conditional
            expectation in the generated partition.
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        f.check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)
        if y is None:
            y = self.y
        prediction_vector = ct.calc_pred(self.selected_rs, y, X)
        return prediction_vector

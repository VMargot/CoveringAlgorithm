import numpy as np
import pandas as pd
from typing import Union, List
from CoveringAlgorithm.rule import Rule
from CoveringAlgorithm import functions as f


class RuleSet(object):
    """
    Class for a ruleset. It's a kind of list of rule object
    """

    def __init__(self, rs):
        if type(rs) in [list, np.ndarray]:
            self.rules = rs
        elif type(rs) == RuleSet:
            self.rules = rs.get_rules()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'ruleset: %s rules' % str(len(self.rules))

    def __gt__(self, val):
        return [rule > val for rule in self.rules]

    def __lt__(self, val):
        return [rule < val for rule in self.rules]

    def __ge__(self, val):
        return [rule >= val for rule in self.rules]

    def __le__(self, val):
        return [rule <= val for rule in self.rules]

    def __add__(self, ruleset):
        return self.extend(ruleset)

    def __getitem__(self, i):
        return self.get_rules()[i]

    def __len__(self):
        return len(self.get_rules())

    def __del__(self):
        if len(self) > 0:
            nb_rules = len(self)
            i = 0
            while i < nb_rules:
                del self[0]
                i += 1

    def __delitem__(self, rules_id):
        del self.rules[rules_id]

    def append(self, rule: Rule):
        """
        Add one rule to a RuleSet object (self).
        """
        assert rule.__class__ == Rule, 'Must be a rule object (try extend)'
        if any(map(lambda r: rule == r, self)) is False:
            self.rules.append(rule)

    def extend(self, ruleset):
        """
        Add rules form a ruleset to a RuleSet object (self).
        """
        assert ruleset.__class__ == RuleSet, 'Must be a ruleset object'
        'ruleset must have the same Learning object'
        rules_list = ruleset.get_rules()
        self.rules.extend(rules_list)
        return self

    def insert(self, idx: int, rule: Rule):
        """
        Insert one rule to a RuleSet object (self) at the position idx.
        """
        assert rule.__class__ == Rule, 'Must be a rule object'
        self.rules.insert(idx, rule)

    def pop(self, idx: int):
        """
        Drop the rule at the position idx.
        """
        self.rules.pop(idx)

    def extract_greater(self, param: str, val: Union[float, int]):
        """
        Extract a RuleSet object from self such as each rules have a param
        greater than val.
        """
        rules_list = list(filter(lambda rule: rule.get_param(param) > val, self))
        return RuleSet(rules_list)

    def extract_least(self, param: str, val: Union[float, int]):
        """
        Extract a RuleSet object from self such as each rules have a param
        least than val.
        """
        rules_list = list(filter(lambda rule: rule.get_param(param) < val, self))
        return RuleSet(rules_list)

    def extract(self, param: str, val: Union[float, int]):
        """
        Extract a RuleSet object from self such as each rules have a param
        equal to val.
        """
        rules_list = list(filter(lambda rule: rule.get_param(param) == val, self))
        return RuleSet(rules_list)

    def extract_length(self, length: int):
        """
        Extract a RuleSet object from self such as each rules have a
        length l.
        """
        return self.extract('length', length)

    def index(self, rule: Rule):
        """
        Get the index a rule in a RuleSet object (self).
        """
        assert rule.__class__ == Rule, 'Must be a rule object'
        self.get_rules().index(rule)

    def replace(self, idx: int, rule: Rule):
        """
        Replace rule at position idx in a RuleSet object (self)
        by a new rule.
        """
        self.rules.pop(idx)
        self.rules.insert(idx, rule)

    def sort_by(self, crit: str, maximized: bool):
        """
        Sort the RuleSet object (self) by a criteria criterion
        """
        self.rules = sorted(self.rules, key=lambda x: x.get_param(crit),
                            reverse=maximized)

    def drop_duplicates(self):
        """
        Drop duplicates rules in RuleSet object (self)
        """
        rules_list = list(set(self.rules))
        return RuleSet(rules_list)

    def to_df(self, cols: List[str] = None):
        """
        To transform an ruleset into a pandas DataFrame
        """
        if cols is None:
            cols = ['Features_Name', 'BMin', 'BMax',
                    'Cov', 'Pred', 'Var', 'Crit', 'Significant']

        df = pd.DataFrame(index=self.get_rules_name(),
                          columns=cols)

        for col_name in cols:
            att_name = col_name.lower()
            if all([hasattr(rule, att_name) for rule in self]):
                df[col_name] = [rule.get_param(att_name) for rule in self]

            elif all([hasattr(rule.conditions, att_name.lower()) for rule in self]):
                df[col_name] = [rule.conditions.get_param(att_name) for rule in self]

        return df

    def calc_pred(self, y_train: np.ndarray, x_train: np.ndarray = None,
                  x_test: np.ndarray = None):
        """
        Computes the prediction vector
        using an rule based partition
        """
        # Activation of all rules in the learning set
        activation_matrix = np.array([rule.get_activation(x_train) for rule in self])

        if x_test is None:
            prediction_matrix = activation_matrix.T
        else:
            prediction_matrix = [rule.calc_activation(x_test) for rule in self]
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

        # Calculation of the expectation of the complementary
        no_act = 1 - self.calc_activation(x_train)
        no_pred = np.mean(np.extract(y_train, no_act))

        # Get empty significant cells
        significant_list = np.array(self.get_rules_param('significant'), dtype=int)
        significant_rules = np.where(significant_list == 1)[0]
        temp = prediction_matrix[:, significant_rules]
        nb_rules_active = temp.sum(axis=1)
        nb_rules_active[nb_rules_active == 0] = -1
        empty_cells = np.where(nb_rules_active == -1)[0]

        # Get empty insignificant cells
        bad_cells = np.where(np.sum(cells, axis=1) == 0)[0]
        bad_cells = list(filter(lambda i: i not in empty_cells, bad_cells))

        # Calculation of the conditional expectation in each cell
        prediction_vector = [f.calc_prediction(act, y_train) for act in cells]
        prediction_vector = np.array(prediction_vector)

        prediction_vector[bad_cells] = no_pred
        prediction_vector[empty_cells] = 0.0

        return prediction_vector, bad_cells, empty_cells

    def calc_activation(self, x: np.ndarray = None):
        """
        Compute the  activation vector of a set of rules
        """
        activation_vector = [rule.get_activation(x) for rule in self]
        activation_vector = np.sum(activation_vector, axis=0)
        activation_vector = 1 * activation_vector.astype('bool')

        return activation_vector

    def calc_coverage(self, x: np.ndarray = None):
        """
        Compute the coverage rate of a set of rules
        """
        if len(self) > 0:
            activation_vector = self.calc_activation(x)
            cov = f.calc_coverage(activation_vector)
        else:
            cov = 0.0
        return cov

    def predict(self, y_train: np.ndarray, x_train: np.ndarray, x_test: np.ndarray):
        """
        Computes the prediction vector for a given x and a given aggregation method
        """
        prediction_vector, bad_cells, no_rules = self.calc_pred(y_train, x_train, x_test)
        return prediction_vector, bad_cells, no_rules

    def make_rule_names(self):
        """
        Add an attribute name at each rule of self
        """
        list(map(lambda rule, rules_id: rule.make_name(rules_id),
                 self, range(len(self))))

    """------   Getters   -----"""
    def get_rules_param(self, param: str):
        """
        To get the list of a parameter param of the rules in self
        """
        return [rule.get_param(param) for rule in self]

    def get_rules_name(self):
        """
        To get the list of the name of rules in self
        """
        try:
            return self.get_rules_param('name')
        except AssertionError:
            self.make_rule_names()
            return self.get_rules_param('name')

    def get_rules(self):
        """
        To get the list of rule in self
        """
        return self.rules

    """------   Setters   -----"""
    def set_rules(self, rules_list):
        """
        To set a list of rule in self
        """
        assert type(rules_list) == list, 'Must be a list object'
        self.rules = rules_list

    def set_rules_cluster(self, params: str, length):
        rules_list = list(filter(lambda rule: rule.get_param('length') == length, self))
        list(map(lambda rule, rules_id: rule.set_params(cluster=params[rules_id]),
                 rules_list, range(len(rules_list))))
        rules_list += list(filter(lambda rule: rule.get_param('length') != length, self))

        self.rules = rules_list

import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from CoveringAlgorithm import functions as rt
from CoveringAlgorithm.ruleconditions import RuleConditions


class Rule(object):
    """
    Class for a rule with a binary rule condition
    """

    def __init__(self,
                 rule_conditions):

        assert rule_conditions.__class__ == RuleConditions, \
            'Must be a RuleCondition object'

        self.conditions = rule_conditions
        self.length = len(rule_conditions.get_param('features_index'))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.conditions == other.conditions

    def __gt__(self, val):
        return self.get_param('pred') > val

    def __lt__(self, val):
        return self.get_param('pred') < val

    def __ge__(self, val):
        return self.get_param('pred') >= val

    def __le__(self, val):
        return self.get_param('pred') <= val

    def __str__(self):
        return 'rule: ' + self.conditions.__str__()

    def __hash__(self):
        return hash(self.conditions)

    def test_included(self, rule, x=None):
        """
        Test to know if a rule (self) and an other (rule)
        are included
        """
        activation_self = self.get_activation(x)
        activation_other = rule.get_activation(x)

        intersection = np.logical_and(activation_self, activation_other)

        if np.allclose(intersection, activation_self) \
                or np.allclose(intersection, activation_other):
            return None
        else:
            return 1 * intersection

    def test_variables(self, rule):
        """
        Test to know if a rule (self) and an other (rule)
        have conditions on the same features.
        """
        c1 = self.conditions
        c2 = rule.conditions

        c1_name = c1.get_param('features_name')
        c2_name = c2.get_param('features_name')
        if len(set(c1_name).intersection(c2_name)) != 0:
            return True
        else:
            return False

    def test_length(self, rule, length):
        """
        Test to know if a rule (self) and an other (rule)
        could be intersected to have a new rule of length length.
        """
        return self.get_param('length') + rule.get_param('length') == length

    def intersect_test(self, rule, x):
        """
        Test to know if a rule (self) and an other (rule)
        could be intersected.

        Test 1: the sum of complexities of self and rule are egal to l
        Test 2: self and rule have not condition on the same variable
        Test 3: self and rule have not included activation
        """
        if self.test_variables(rule) is False:
            return self.test_included(rule=rule, x=x)
        else:
            return None

    def union_test(self, activation, gamma=0.80, x=None):
        """
        Test to know if a rule (self) and an activation vector have
        at more gamma percent of points in common
        """
        self_vect = self.get_activation(x)
        intersect_vect = np.logical_and(self_vect, activation)

        pts_inter = np.sum(intersect_vect)
        pts_rule = np.sum(activation)
        pts_self = np.sum(self_vect)

        ans = (pts_inter < gamma * pts_self) and (pts_inter < gamma * pts_rule)

        return ans

    def intersect_conditions(self, rule):
        """
        Compute an RuleCondition object from the intersection of an rule
        (self) and an other (rulessert)
        """
        conditions_1 = self.conditions
        conditions_2 = rule.conditions

        conditions = list(map(lambda c1, c2: c1 + c2, conditions_1.get_attr(),
                              conditions_2.get_attr()))

        return conditions

    def intersect(self, rule, cov_min, cov_max, x, low_memory):
        """
        Compute a suitable rule object from the intersection of an rule
        (self) and an other (rulessert).
        Suitable means that self and rule satisfied the intersection test
        """
        new_rule = None
        # if self.get_param('pred') * rule.get_param('pred') > 0:
        activation = self.intersect_test(rule, x)
        if activation is not None:
            cov = rt.calc_coverage(activation)
            if cov_min <= cov <= cov_max:
                conditions_list = self.intersect_conditions(rule)

                new_conditions = RuleConditions(features_name=conditions_list[0],
                                                features_index=conditions_list[1],
                                                bmin=conditions_list[2],
                                                bmax=conditions_list[3],
                                                xmax=conditions_list[5],
                                                xmin=conditions_list[4])
                new_rule = Rule(new_conditions)
                if low_memory is False:
                    new_rule.set_params(activation=activation)

        return new_rule

    def calc_stats(self, x, y, method='mse',
                   cov_min=0.01, cov_max=0.5, low_memory=False):
        """
        Calculation of all statistics of an rules

        Parameters
        ----------
        x : {array-like or discretized matrix, shape = [n, d]}
            The training input samples after discretization.

        y : {array-like, shape = [n]}
            The normalized target values (real numbers).

        method : {string type}
                 The method mse_function or msecriterion

        cov_min : {float type such as 0 <= covmin <= 1}, default 0.5
                  The minimal coverage of one rule

        cov_max : {float type such as 0 <= covmax <= 1}, default 0.5
                  The maximal coverage of one rule

        low_memory : {bool type}
                     To save activation vectors of rules

        Return
        ------
        None : if the rule does not verified coverage conditions
        """
        self.set_params(out=False)
        activation_vector = self.calc_activation(x=x)

        if sum(activation_vector) > 0:
            self.set_params(activation=activation_vector)

            y_fillna = np.nan_to_num(y)
            y_cond = np.extract(activation_vector, y_fillna)

            cov = rt.calc_coverage(activation_vector)
            self.set_params(cov=cov)

            # prediction = rt.calc_prediction(activation_vector, y_fillna)
            self.set_params(pred=np.mean(y_cond))

            # cond_var = rt.calc_variance(activation_vector, y)
            self.set_params(var=np.var(y_cond))

            rez = rt.calc_criterion(self.get_param('pred'), y_cond, method)
            self.set_params(crit=rez)

        else:
            print('No activation for rule %s' % (str(self)))

    def calc_activation(self, x=None):
        """
        Compute the activation vector of an rule
        """
        return self.conditions.transform(x)

    def predict(self, x=None):
        """
        Compute the prediction of an rule
        """
        prediction = self.get_param('pred')
        if x is not None:
            activation = self.calc_activation(x=x)
        else:
            activation = self.get_activation()

        return prediction * activation

    def score(self, x, y, sample_weight=None, score_type='Rate'):
        """
        Returns the coefficient of determination R^2 of the prediction
        if y is continuous. Else if y in {0,1} then Returns the mean
        accuracy on the given test data and labels {0,1}.

        Parameters
        ----------
        x : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for x.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        score_type : string-type

        Returns
        -------
        score : float
            R^2 of self.predict(x) wrt. y in R.

            or

        score : float
            Mean accuracy of self.predict(x) wrt. y in {0,1}
        """
        prediction_vector = self.predict(x)

        y = np.extract(np.isfinite(y), y)
        prediction_vector = np.extract(np.isfinite(y), prediction_vector)

        if score_type == 'Classification':
            th_val = (min(y) + max(y)) / 2.0
            prediction_vector = list(map(lambda p: min(y) if p < th_val else max(y),
                                         prediction_vector))
            return accuracy_score(y, prediction_vector)

        elif score_type == 'Regression':
            return r2_score(y, prediction_vector, sample_weight=sample_weight,
                            multioutput='variance_weighted')

    def make_name(self, num, learning=None):
        """
        Add an attribute name to self

        Parameters
        ----------
        num : int
              index of the rule in an ruleset

        learning : Learning object, default None
                   If leaning is not None the name of self will
                   be defined with the name of learning
        """
        name = 'R ' + str(num)
        length = self.get_param('length')
        name += '(' + str(length) + ')'
        prediction = self.get_param('pred')
        if prediction > 0:
            name += '+'
        elif prediction < 0:
            name += '-'

        if learning is not None:
            dtstart = learning.get_param('dtstart')
            dtend = learning.get_param('dtend')
            if dtstart is not None:
                name += str(dtstart) + ' '
            if dtend is not None:
                name += str(dtend)

        self.set_params(name=name)

    """------   Getters   -----"""

    def get_param(self, param):
        """
        To get the parameter param
        """
        assert type(param) == str, 'Must be a string'
        assert hasattr(self, param), \
            'self.%s must be calculate before' % param
        return getattr(self, param)

    def get_activation(self, x=None):
        """
        To get the activation vector of self.
        If it does not exist the function return None
        """
        if x is not None:
            return self.conditions.transform(x)
        else:
            if hasattr(self, 'activation'):
                return self.get_param('activation')
            else:
                print('No activation vector for %s' % str(self))
            return None

    def get_predictions_vector(self, x=None):
        """
        To get the activation vector of self.
        If it does not exist the function return None
        """
        if hasattr(self, 'pred'):
            prediction = self.get_param('pred')
            if hasattr(self, 'activation'):
                return prediction * self.get_param('activation')
            else:
                return prediction * self.calc_activation(x)
        else:
            return None

    """------   Setters   -----"""

    def set_params(self, **parameters):
        """
        To set a new parameter
        Example:
        --------
        o.set_params(new_param=val_new_param)
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class RuleConditions(object):
    """
    Class for binary rule condition
    """

    def __init__(self, features_name, features_index,
                 bmin, bmax, xmin, xmax, values=None):

        assert isinstance(features_name, (tuple, list, np.ndarray)), \
            'Type of parameter must be iterable tuple, list or array' % features_name
        self.features_name = features_name
        length = len(features_name)

        assert isinstance(features_index, (tuple, list, np.ndarray)), \
            'Type of parameter must be iterable tuple, list or array' % features_name
        assert len(features_index) == length, \
            'Parameters must have the same length' % features_name
        self.features_index = features_index

        assert isinstance(bmin, (tuple, list, np.ndarray)), \
            'Type of parameter must be iterable tuple, list or array' % features_name
        assert len(bmin) == length, \
            'Parameters must have the same length' % features_name
        assert isinstance(bmax, (tuple, list, np.ndarray)), \
            'Type of parameter must be iterable tuple, list or array' % features_name
        assert len(bmax) == length, \
            'Parameters must have the same length' % features_name
        if type(bmin[0]) != str:
            assert all(map(lambda a, b: a <= b, bmin, bmax)), \
                'Bmin must be smaller or equal than bmax (%s)' \
                % features_name
        self.bmin = bmin
        self.bmax = bmax

        assert isinstance(xmax, (tuple, list, np.ndarray)), \
            'Type of parameter must be iterable tuple, list or array' % features_name
        assert len(xmax) == length, \
            'Parameters must have the same length' % features_name
        assert isinstance(xmin, (tuple, list, np.ndarray)), \
            'Type of parameter must be iterable tuple, list or array' % features_name
        assert len(xmin) == length, \
            'Parameters must have the same length' % features_name
        self.xmin = xmin
        self.xmax = xmax

        if values is None:
            values = []
        else:
            assert isinstance(values, (tuple, list, np.ndarray)), \
                'Type of parameter must be iterable tuple, list or array' % features_name

        self.values = [values]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        features = self.features_name
        return "Var: %s, Bmin: %s, Bmax: %s" % (features, self.bmin, self.bmax)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        to_hash = [(self.features_index[i], self.features_name[i],
                    self.bmin[i], self.bmax[i])
                   for i in range(len(self.features_index))]
        to_hash = frozenset(to_hash)
        return hash(to_hash)

    def transform(self, X):
        """
        Transform a matrix xmat into an activation vector.
        It means an array of 0 and 1. 0 if the condition is not
        satisfied and 1 otherwise.

        Parameters
        ----------
        X: {array-like matrix, shape=(n_samples, n_features)}
              Input data

        Returns
        -------
        activation_vector: {array-like matrix, shape=(n_samples, 1)}
                     The activation vector
        """
        length = len(self.features_name)
        geq_min = True
        leq_min = True
        not_nan = True
        for i in range(length):
            col_index = self.features_index[i]
            x_col = X[:, col_index]

            # Turn x_col to array
            if len(x_col) > 1:
                x_col = np.squeeze(np.asarray(x_col))

            if type(self.bmin[i]) == str:
                x_col = np.array(x_col, dtype=np.str)

                temp = (x_col == self.bmin[i])
                temp |= (x_col == self.bmax[i])
                geq_min &= temp
                leq_min &= True
                not_nan &= True
            else:
                x_col = np.array(x_col, dtype=np.float)

                x_temp = [self.bmin[i] - 1 if x != x else x for x in x_col]
                geq_min &= np.greater_equal(x_temp, self.bmin[i])

                x_temp = [self.bmax[i] + 1 if x != x else x for x in x_col]
                leq_min &= np.less_equal(x_temp, self.bmax[i])

                not_nan &= np.isfinite(x_col)

        activation_vector = 1 * (geq_min & leq_min & not_nan)

        return activation_vector

    """------   Getters   -----"""

    def get_param(self, param):
        """
        To get the parameter param
        """
        assert type(param) == str, \
            'Must be a string'

        return getattr(self, param)

    def get_attr(self):
        """
        To get a list of attributes of self.
        It is useful to quickly create a RuleConditions
        from intersection of two rules
        """
        return [self.features_name,
                self.features_index,
                self.bmin, self.bmax,
                self.xmin, self.xmax]

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

    def intersect_test(self, rule, X):
        """
        Test to know if a rule (self) and an other (rule)
        could be intersected.

        Test 1: the sum of complexities of self and rule are egal to l
        Test 2: self and rule have not condition on the same variable
        Test 3: self and rule have not included activation
        """
        if self.test_variables(rule) is False:
            return self.test_included(rule=rule, x=X)
        else:
            return None

    def union_test(self, activation, gamma=0.80, X=None):
        """
        Test to know if a rule (self) and an activation vector have
        at more gamma percent of points in common
        """
        self_vect = self.get_activation(X)
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

    def intersect(self, rule, cov_min, cov_max, X, low_memory):
        """
        Compute a suitable rule object from the intersection of an rule
        (self) and an other (rulessert).
        Suitable means that self and rule satisfied the intersection test
        """
        new_rule = None
        # if self.get_param('pred') * rule.get_param('pred') > 0:
        activation = self.intersect_test(rule, X)
        if activation is not None:
            cov = calc_coverage(activation)
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
            if low_memory is False:
                self.set_params(activation=activation_vector)

            cov = calc_coverage(activation_vector)
            self.set_params(cov=cov)

            if cov >= cov_max or cov <= cov_min:
                self.set_params(out=True)
            else:
                prediction = calc_prediction(activation_vector, y)
                self.set_params(pred=prediction)

                cond_var = calc_variance(activation_vector, y)
                self.set_params(var=cond_var)

                prediction_vector = activation_vector * prediction
                complementary_prediction = calc_prediction(1 - activation_vector, y)
                np.place(prediction_vector, prediction_vector == 0,
                         complementary_prediction)

                rez = calc_criterion(prediction_vector, y, method)
                self.set_params(crit=rez)

        else:
            self.set_params(out=True)

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
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        score_type : string-type

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y in R.

            or

        score : float
            Mean accuracy of self.predict(X) wrt. y in {0,1}
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

    def append(self, rule):
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

    def insert(self, idx, rule):
        """
        Insert one rule to a RuleSet object (self) at the position idx.
        """
        assert rule.__class__ == Rule, 'Must be a rule object'
        self.rules.insert(idx, rule)

    def pop(self, idx=None):
        """
        Drop the rule at the position idx.
        """
        self.rules.pop(idx)

    def extract_greater(self, param, val):
        """
        Extract a RuleSet object from self such as each rules have a param
        greater than val.
        """
        rules_list = list(filter(lambda rule: rule.get_param(param) > val, self))
        return RuleSet(rules_list)

    def extract_least(self, param, val):
        """
        Extract a RuleSet object from self such as each rules have a param
        least than val.
        """
        rules_list = list(filter(lambda rule: rule.get_param(param) < val, self))
        return RuleSet(rules_list)

    def extract_length(self, length):
        """
        Extract a RuleSet object from self such as each rules have a
        length l.
        """
        rules_list = list(filter(lambda rule: rule.get_param('length') == length, self))
        return RuleSet(rules_list)

    def extract(self, param, val):
        """
        Extract a RuleSet object from self such as each rules have a param
        equal to val.
        """
        rules_list = list(filter(lambda rule: rule.get_param(param) == val, self))
        return RuleSet(rules_list)

    def index(self, rule):
        """
        Get the index a rule in a RuleSet object (self).
        """
        assert rule.__class__ == Rule, 'Must be a rule object'
        self.get_rules().index(rule)

    def replace(self, idx, rule):
        """
        Replace rule at position idx in a RuleSet object (self)
        by a new rule.
        """
        self.rules.pop(idx)
        self.rules.insert(idx, rule)

    def sort_by(self, crit, maximized):
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

    def to_df(self, cols=None):
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

    def calc_pred(self, y_train, x_train=None, x_test=None):
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
        prediction_vector = [calc_prediction(act, y_train) for act in cells]
        prediction_vector = np.array(prediction_vector)

        prediction_vector[bad_cells] = no_pred
        prediction_vector[empty_cells] = 0.0

        return prediction_vector, bad_cells, empty_cells

    def calc_activation(self, x=None):
        """
        Compute the  activation vector of a set of rules
        """
        activation_vector = [rule.get_activation(x) for rule in self]
        activation_vector = np.sum(activation_vector, axis=0)
        activation_vector = 1 * activation_vector.astype('bool')

        return activation_vector

    def calc_coverage(self, x=None):
        """
        Compute the coverage rate of a set of rules
        """
        if len(self) > 0:
            activation_vector = self.calc_activation(x)
            cov = calc_coverage(activation_vector)
        else:
            cov = 0.0
        return cov

    def predict(self, y_train, x_train, x_test):
        """
        Computes the prediction vector for a given X and a given aggregation method
        """
        prediction_vector, bad_cells, no_rules = self.calc_pred(y_train, x_train, x_test)
        return prediction_vector, bad_cells, no_rules

    def make_rule_names(self):
        """
        Add an attribute name at each rule of self
        """
        list(map(lambda rule, rules_id: rule.make_name(rules_id),
                 self, range(len(self))))

    def make_selected_df(self):
        df = self.to_df()

        df.rename(columns={"Cov": "Coverage", "Pred": "Prediction",
                           'Var': 'Variance', 'Crit': 'Criterion'},
                  inplace=True)

        df['Conditions'] = [make_condition(rule) for rule in self]
        selected_df = df[['Conditions', 'Coverage',
                          'Prediction', 'Variance',
                          'Criterion']].copy()

        selected_df['Coverage'] = selected_df.Coverage.round(2)
        selected_df['Prediction'] = selected_df.Prediction.round(2)
        selected_df['Variance'] = selected_df.Variance.round(2)
        selected_df['Criterion'] = selected_df.Criterion.round(2)

        return selected_df

    def plot_counter_variables(self, nb_max=None):
        counter = get_variables_count(self)

        x_labels = list(map(lambda item: item[0], counter))
        values = list(map(lambda item: item[1], counter))

        f = plt.figure()
        ax = plt.subplot()

        if nb_max is not None:
            x_labels = x_labels[:nb_max]
            values = values[:nb_max]

        g = sns.barplot(y=x_labels, x=values, ax=ax, ci=None)
        g.set(xlim=(0, max(values) + 1), ylabel='Variable', xlabel='Count')

        return f

    def plot_dist(self, x=None, metric=dist):
        rules_names = self.get_rules_name()

        predictions_vector_list = [rule.get_predictions_vector(x) for rule in self]
        predictions_matrix = np.array(predictions_vector_list)

        distance_vector = scipy_dist.pdist(predictions_matrix, metric=metric)
        distance_matrix = scipy_dist.squareform(distance_vector)

        # Set up the matplotlib figure
        f = plt.figure()
        ax = plt.subplot()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(distance_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        vmax = np.max(distance_matrix)
        vmin = np.min(distance_matrix)
        # center = np.mean(distance_matrix)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(distance_matrix, cmap=cmap, ax=ax,
                    vmax=vmax, vmin=vmin, center=1.,
                    square=True, xticklabels=rules_names,
                    yticklabels=rules_names, mask=mask)

        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        return f

    """------   Getters   -----"""
    def get_candidates(self, X, k, length, method, nb_jobs):
        candidates = []
        for l in [1, length - 1]:
            rs_length_l = self.extract_length(l)
            if method == 'cluter':
                if all(map(lambda rule: hasattr(rule, 'cluster'),
                           rs_length_l)) is False:
                    clusters = find_cluster(rs_length_l,
                                            X, k, nb_jobs)
                    self.set_rules_cluster(clusters, l)

                rules_list = []
                for i in range(k):
                    sub_rs = rs_length_l.extract('cluster', i)
                    if len(sub_rs) > 0:
                        sub_rs.sort_by('var', True)
                        rules_list.append(sub_rs[0])

            elif method == 'best':
                rs_length_l.sort_by('crit', False)
                rules_list = rs_length_l[:k]

            else:
                print('Choose a method among [cluster, best] to select candidat')
                rules_list = rs_length_l.rules

            candidates.append(RuleSet(rules_list))

        return candidates[0], candidates[1]

    def get_rules_param(self, param):
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

    def set_rules_cluster(self, params, length):
        rules_list = list(filter(lambda rule: rule.get_param('length') == length, self))
        list(map(lambda rule, rules_id: rule.set_params(cluster=params[rules_id]),
                 rules_list, range(len(rules_list))))
        rules_list += list(filter(lambda rule: rule.get_param('length') != length, self))

        self.rules = rules_list

from CoveringAlgorithm import CA
from sklearn.datasets import load_diabetes, load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np


def make_y(x, noise, th_min=-0.4, th_max=0.4):
    y_vect = [-2 if x_val <= th_min else 0 if x_val <= th_max else 2 for x_val in x]
    y_vect += np.random.normal(0, noise, len(y_vect))
    return np.array(y_vect)


if __name__ == "__main__":
    # Covering parameters
    alpha = 1.0 / 2 - 1.0 / 100
    gamma = 0.90
    lmax = 3
    learning_rate = 0.1
    max_depth = 3  # number of level by tree
    tree_size = 2 ** max_depth  # number of leaves by tree
    max_rules = 4000  # total number of rules generated from tree ensembles

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    ca_rf = CA.CA(
        alpha=alpha,
        gamma=gamma,
        tree_size=tree_size,
        max_rules=max_rules,
        generator_func=RandomForestRegressor,
        lmax=lmax,
        seed=42,
    )
    ca_rf.fit(X_train, y_train)
    pred = ca_rf.predict(X_test)
    print('Boston % of bad points: ', sum(1 - np.isfinite(pred)) / len(pred) * 100)
    pred = np.nan_to_num(pred)
    print('Boston: ', r2_score(y_test, pred))
    print('Boston sigma 2:', min([rule.std**2 for rule in ca_rf.selected_rs]))
    # Boston:  0.44632846823132255

    print("")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    ca_rf = CA.CA(
        alpha=alpha,
        gamma=gamma,
        tree_size=tree_size,
        max_rules=max_rules,
        generator_func=RandomForestRegressor,
        lmax=lmax,
        seed=42,
    )

    ca_rf.fit(X_train, y_train)
    pred = ca_rf.predict(X_test)
    pred = np.nan_to_num(pred)
    print("Diabetes: ", r2_score(y_test, pred))
    print("Diabetes sigma 2:", min([rule.std ** 2 for rule in ca_rf.selected_rs]))
    # Diabetes: 0.4113726356207583
    # Diabetes sigma 2: 2428.159722222222

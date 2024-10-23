"""
Configuration file with constants used in other modules
"""
import numpy as np

RAND_VAL = 3001

gam_params = {
    "interactions": range(0, 6),
    "learning_rate": np.linspace(0.0005, 0.02, 5),
    "max_leaves": range(3, 6),
    "random_state": [RAND_VAL]
}

gosdt_params = {
    "depth_budget": range(5, 11),
    "regularization": np.linspace(0.005, 0.1, 5),
    "balance": [True, False],
    "random_state": [RAND_VAL]
}

slim_params = {
    "C_0": [0.001, 0.0025, 0.005, 0.0075, 0.01]
}

cat_params = {
    "num_trees": [x for x in range(20, 61, 5)],
    "learning_rate": [x for x in np.linspace(0.0005, 0.02, 5)],
    "max_depth": [x for x in range(4, 12)],
    "verbose": [False],
    "random_state": [RAND_VAL]
}
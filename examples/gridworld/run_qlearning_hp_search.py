from rlpy.Tools.hypersearch import find_hyperparameters
best, trials = find_hyperparameters(
    "examples/gridworld/qlearning_hyperparamsearch.py",
    "./Results/gridworld/qlearning_hpsearch",
    max_evals=10, parallelization="joblib",
    trials_per_point=5)
print best

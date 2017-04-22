#!/usr/bin/env python

__author__ = "William Dabney"

from rlpy.Domains import GridWorld
from rlpy.Agents import Q_Learning
from rlpy.Representations import iFDDK, IndependentDiscretization
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp
import os


param_space = {
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1)),
    'epsilon': hp.loguniform("epsilon", np.log(5e-2), np.log(1))}


def make_experiment(exp_id=1, path="./Results/Temp",
                    lambda_ = 0.3,
                    boyan_N0 = 100,
                    initial_learn_rate = 0.11,
                    epsilon = 0.1):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    # Experiment variables
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 1000000
    opt["num_policy_checks"] = 20

    # Logging

    # Domain:
    # MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    maze = os.path.join(GridWorld.default_map_dir, '4x5.txt')
    domain = GridWorld(maze, noise=0.3)
    opt["domain"] = domain

    initial_rep = IndependentDiscretization(domain)
    representation = iFDDK(domain, 1., initial_rep,
                          sparsify=True,
                          useCache=True, lazy=True,
                          lambda_=lambda_)

    # Policy
    policy = eGreedy(representation, epsilon=epsilon)

    # Agent
    opt["agent"] = Q_Learning(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)

    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    path = "./Results/Temp/{domain}/{agent}/{representation}/"
    experiment = make_experiment(1, path=path)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=True,  # show performance runs?
                   visualize_performance=True)  # show value function?
    experiment.plot()
    experiment.save()

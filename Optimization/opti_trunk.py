# -*- coding: utf-8 -*-
import sys
import os
import time
import random as rd
import gym
import json
from scipy import stats
import optuna
import multiprocessing as mp
import numpy as np
import joblib
from tools_opti import data_separation

__import__('sofagym')
sys.path.insert(0, os.getcwd()+"/..")


ENV_NAME = 'abstracttrunkcube-v0'
with open("./Results/optimisation/abstracttrunk/real_points.txt", 'r') as outfile:
    pos = json.load(outfile)
    n_pos = len(pos)
    print(">>   Number of position:", n_pos)
    print(">>   Number of points:", len(pos[0]))
    TRAIN_POINTS, VALIDATION_POINTS = [], []
    for i, p in enumerate(pos):
        if (i+1)%5==0:
            TRAIN_POINTS.append(p)
        else:
            VALIDATION_POINTS.append(p)

PATH = "./Results/optimisation/abstracttrunk/"
os.makedirs(PATH, exist_ok = True)

def evaluate(args):
    len_beam= args["len_beam"]
    max_flex_1 = args["max_flex_1"]
    max_flex_2= args["max_flex_2"]
    young_modulus_1 = args["young_modulus_1"]
    young_modulus_2= args["young_modulus_2"]
    mass_left= args["mass_left"]
    mass_right = args["mass_right"]
    rest_force = args["rest_force"]

    print("Start env ", ENV_NAME)

    env = gym.make(ENV_NAME)
    env.configure({ "len_beam": len_beam,
        "max_flex": [max_flex_1, max_flex_2],
        "young_modulus": [young_modulus_1, young_modulus_2],
        "mass": [mass_left, mass_right],
        "rest_force": rest_force,
        "render":0,
        "scale_factor":30,
        "dt":0.01})

    env.reset()

    print("Start ...")
    _state_1, _, _, _ = env.step([0, -1, 0, 0, -1])
    _state_2, _, _, _ = env.step([0, 0, 0, -0.5, -1])
    _state_3, _, _, _ = env.step([0, 1, 0, -1, -1])

    _state = _state_1.tolist()+_state_2.tolist()+_state_3.tolist()
    points = []
    for i in range(n_pos):
        if (i+1)%5==0:
            points.append(_state[i])

    sum = 0
    nb_points = len(points)
    for i in range(nb_points):
        p_real, p_proxy = np.array(TRAIN_POINTS[i]), np.array(points[i])
        sum+= np.linalg.norm(p_real-p_proxy, axis = 1).mean()/nb_points

    return sum


def validate(args):
    len_beam= args["len_beam"]
    max_flex_1 = args["max_flex_1"]
    max_flex_2= args["max_flex_2"]
    young_modulus_1 = args["young_modulus_1"]
    young_modulus_2= args["young_modulus_2"]
    mass_left= args["mass_left"]
    mass_right = args["mass_right"]
    rest_force = args["rest_force"]


    print("Start env ", ENV_NAME)

    env = gym.make(ENV_NAME)
    env.configure({ "len_beam": len_beam,
        "max_flex": [max_flex_1, max_flex_2],
        "young_modulus": [young_modulus_1, young_modulus_2],
        "mass": [mass_left, mass_right],
        "rest_force": rest_force,
        "render":0,
        "scale_factor":30,
        "dt":0.01})

    env.reset()

    print("Start ...")
    _state_1, _, _, _ = env.step([0, -1, 0, 0, -1])
    _state_2, _, _, _ = env.step([0, 0, 0, -0.5, -1])
    _state_3, _, _, _ = env.step([0, 1, 0, -1, -1])

    _state = _state_1.tolist()+_state_2.tolist()+_state_3.tolist()
    points = []
    for i in range(n_pos):
        if (i+1)%5!=0:
            points.append(_state[i])


    sum, coord_sum = 0, np.array([0.,0.,0.])
    nb_points = len(points)
    for i in range(nb_points):
        p_real, p_proxy = np.array(VALIDATION_POINTS[i]), np.array(points[i])
        sum+= np.linalg.norm(p_real-p_proxy, axis = 1).mean()/nb_points
        coord_sum+= np.mean(np.abs(p_real-p_proxy), axis = 0)/nb_points
    print(">>  ... Done with score: ", sum, coord_sum)
    return sum

def objective(trial):
    params = {"len_beam": trial.suggest_uniform('len_beam', 192, 197),
              "young_modulus_1": trial.suggest_uniform('young_modulus_1', 1e2, 1e6),
              "young_modulus_2": trial.suggest_uniform('young_modulus_2', 1e2, 1e6),
              "max_flex_1": trial.suggest_uniform('max_flex_1', 0.005, 0.05),
              "max_flex_2": trial.suggest_uniform('max_flex_2', 0.005, 0.05),
              "mass_left": trial.suggest_uniform('mass_left', 5, 10),
              "mass_right": trial.suggest_uniform('mass_right',0, 5),
              "rest_force":trial.suggest_uniform('rest_force', 0, 1e5),
            }
    return evaluate(params)


# print(">>> START OPTIMISATION ... ")
# n_cores = 1 #mp.cpu_count()
# study = optuna.create_study(direction='minimize')
# # study=joblib.load("./Results/optimisation/abstracttrunk/study.pkl")
# validation, epoch = [], []
# for i in range(1000):
#     study.optimize(objective, n_jobs=n_cores, n_trials=20*n_cores, timeout=None)
#     joblib.dump(study, "./Results/optimisation/abstracttrunk/study.pkl")
#     print(">>  Best Params ", study.best_params)
#     if i%5==0:
#         v= validate(study.best_params)
#         validation.append(v)
#         epoch.append(i*20*n_cores)
#
#         with open(PATH + "validation.txt", 'w') as fp:
#             json.dump([validation, epoch], fp)
#
# print("\n>>  RESULTS:")
# print(">>  Study ", study)
# print(">>  Best Params ", study.best_params)

# TO LOAD:
#
study=joblib.load("./Results/optimisation/abstracttrunk/study.pkl")

print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

v= validate(study.best_params)

fig = optuna.visualization.plot_slice(study)
fig.show()

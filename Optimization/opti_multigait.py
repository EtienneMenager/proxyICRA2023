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

__import__('sofagym')
sys.path.insert(0, os.getcwd()+"/..")

ENV_NAME = 'optiabstractmultigait-v0'
PATH = "./Proxy/optimisation/"+ENV_NAME+"/"
FREQ_TRAIN = 5
os.makedirs(PATH, exist_ok = True)


### Load data
with open(PATH+"ref_points.txt", 'r') as outfile:
    data = json.load(outfile)
print(">>   Number of position:", len(data["data_point"]))
for i in range(len(data["data_point"])):
    print(">>   Number of points for position ", i, ":", len(data["data_point"][i]))

BARYCENTRE_MODEL = np.array(data["barycentre"])
BARYCENTRE_ABSTRACT = np.array([21.625910918673895, 0.0883752212118508, -18.499154748530753])
TRANSLATION = BARYCENTRE_ABSTRACT- BARYCENTRE_MODEL
TRAIN_POINTS, VALIDATION_POINTS = [], []
for pos in data["data_point"]:
    for i, p in enumerate(pos):
        if (i+1)%FREQ_TRAIN==0:
            TRAIN_POINTS.append(p)
        else:
            VALIDATION_POINTS.append(p)
TRAIN_POINTS, VALIDATION_POINTS = np.array(TRAIN_POINTS)+TRANSLATION, np.array(VALIDATION_POINTS)+TRANSLATION

ACTIONS = [[-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
            [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0, -1.0], [1.0, -1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]


def evaluate(args):
    young_modulus_leg= args["young_modulus_leg"]
    young_modulus_center= args["young_modulus_center"]
    mass_density_leg= args["mass_density_leg"]
    mass_density_center= args["mass_density_center"]
    def_cent= args["def_cent"]
    def_leg= args["def_leg"]

    print("Start env ", ENV_NAME)
    env = gym.make(ENV_NAME)
    env.configure({ "young_modulus": [young_modulus_leg, young_modulus_center],
        "mass_density": [mass_density_leg, mass_density_center],
        "def_cent": def_cent,
        "def_leg": def_leg,
        "render":0,
        "dt":0.01,
        "time_before_start": 20})
    env.reset()

    print("Start ...")
    points = []

    for i, action in enumerate(ACTIONS):
        print("\n--------------------------------")
        print("Iteration: ", i, " - action: ", action)
        print("--------------------------------\n")

        _state, _, _, _ = env.step(action)
        state = _state.tolist()
        if i == 0:
            barycentre = np.array(state[0]).mean(axis = 0)
        for i, p in enumerate(state):
            if (i+1)%FREQ_TRAIN==0:
                points.append(p)

    sum = 0
    n_points = len(points)

    for i in range(n_points):
        p_real, p_proxy = np.array(TRAIN_POINTS[i]), np.array(points[i])
        sum+= np.linalg.norm(p_real-p_proxy, axis = 1).mean()/n_points

    return sum


def validate(args):
    young_modulus_leg= args["young_modulus_leg"]
    young_modulus_center= args["young_modulus_center"]
    mass_density_leg= args["mass_density_leg"]
    mass_density_center= args["mass_density_center"]
    def_cent= args["def_cent"]
    def_leg= args["def_leg"]

    print(">>  Start validation with ", ENV_NAME, "...")

    env = gym.make(ENV_NAME)
    env.configure({
        "young_modulus": [young_modulus_leg, young_modulus_center],
        "mass_density": [mass_density_leg, mass_density_center],
        "def_cent": def_cent,
        "def_leg": def_leg,
        "render":0,
        "dt":0.01,
        "time_before_start": 20})

    env.reset()


    print("Start ...")
    points = []
    for i, action in enumerate(ACTIONS):
        _state, _, _, _ = env.step(action)
        state = _state.tolist()

        for i, p in enumerate(state):
            if (i+1)%FREQ_TRAIN!=0:
                points.append(p)

    sum, coord_sum = 0, np.array([0.,0.,0.])
    n_points = len(points)
    for i in range(n_points):
        p_real, p_proxy = np.array(VALIDATION_POINTS[i]), np.array(points[i])
        sum+= np.linalg.norm(p_real-p_proxy, axis = 1).mean()/n_points
        coord_sum+= np.mean(np.abs(p_real-p_proxy), axis = 0)/n_points
    print(">>  ... Done with score: ", sum, coord_sum)
    return sum

def objective(trial):
    params = {"young_modulus_leg": trial.suggest_uniform('young_modulus_leg', 3900, 4100),
              "young_modulus_center": trial.suggest_uniform('young_modulus_center', 1200, 1400),
              "mass_density_leg": trial.suggest_uniform('mass_density_leg', 2e-5, 5e-5),
              "mass_density_center": trial.suggest_uniform('mass_density_center', 9e-8, 2e-7),
              "def_cent": trial.suggest_uniform('def_leg', 0.041, 0.043),
              "def_leg": trial.suggest_uniform('def_cent', 0.019, 0.022)}
    return evaluate(params)


#
# print(">>> START OPTIMISATION ... ")
# n_cores = 1 #mp.cpu_count()
# study = optuna.create_study(direction='minimize')
# # study=joblib.load(PATH+"study.pkl")
# validation, epoch = [], []
# for i in range(1000):
#     study.optimize(objective, n_jobs=n_cores, n_trials=20*n_cores, timeout=None)
#     joblib.dump(study, PATH+"study.pkl")
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

#TO LOAD:
study=joblib.load(PATH+"study.pkl")

print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")

params = dict(study.best_trial.params.items())
print("\"young_modulus\": [{}, {}],".format(params["young_modulus_leg"], params["young_modulus_center"]))
print("\"mass_density\": [{}, {}],".format(params["mass_density_leg"], params["mass_density_center"]))
print("\"def_leg\": {},".format(params["def_leg"]))
print("\"def_cent\": {},".format(params["def_cent"]))

v= validate(study.best_params)
# fig = optuna.visualization.plot_slice(study)
# # fig.show()

# -*- coding: utf-8 -*-
"""Client to run sofa scene.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@inria.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2022, Inria"
__date__ = "Jul 11 2022"

import xmlrpc.client

from os.path import dirname, abspath
import numpy as np
import os
import sys
import ast
import json
import time
import random
import importlib
import gym

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../../../")
command = None
from http.client import BadStatusLine

def load_environment(id, rank, seed = 0):
    def _init():
        __import__('sofagym')
        env = gym.make(id)
        env.seed(seed + rank)
        return env

    return _init

def step(env, action, client):
    """Compute one simulation step.
    """
    obs, reward, _, _ = env.step(action)
    infos = env.get_infos()

    #Notify the server that the task is done
    handle = False
    while not handle:
        try:
            client.taskDone("sofagym", make_result(obs.tolist(), float(reward), infos))
            handle = True
        except BadStatusLine:
            print("[ERROR]  BadStatusLine - rescend the request")

def make_result(obs, reward, infos):
    """Put the result in the right format.
    """

    return {"observation": obs,
            "reward": float(reward),
            "infos": infos}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("SYNTAX: python3.9 client_proxy.py port_rpc env_name")
        sys.exit(-1)

    port_rpc = sys.argv[1]
    env_name = sys.argv[2]


    with xmlrpc.client.ServerProxy('http://localhost:'+ port_rpc) as client:
        env = load_environment(env_name, rank =0, seed = 0)()
        env.configure({"render":0})
        env.configure({"visuQP":False})

        obs = env.reset()
        handle = False
        while not handle:
            try:
                client.registerFirstObservation("sofagym", make_result(obs.tolist(), 0.0, False))
                handle = True
            except BadStatusLine:
                print("[ERROR]  BadStatusLine - rescend the request")

        command = None
        while command != "exit":
            handle = False
            while not handle:
                try:
                    task = client.getNextTask("sofagym")
                    handle = True
                except BadStatusLine:
                    print("[ERROR]  BadStatusLine - rescend the request")

            command = task["command"]
            if command == "step":
                step(env, task["action"], client)

            if command == "setInfos":
                env.set_infos(task["infos"])

    #Avoid that all client exit in the same time
    time.sleep(random.random())
    print("[INFO] >> Client is closed. Bye bye.")

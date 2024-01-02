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

import Sofa
import SofaRuntime

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../../../")
command = None
from http.client import BadStatusLine

def init_simulation(config):
    inverse_scene_path = config['inverse_scene_path']
    create_scene = importlib.import_module(inverse_scene_path).createScene

    root = Sofa.Core.Node("root")
    SofaRuntime.importPlugin("SofaComponentAll")
    create_scene(root, config)
    Sofa.Simulation.init(root)
    return root

def step(root, scale, dt,  client):
    for i in range(scale):
         Sofa.Simulation.animate(root, dt)
    infos = root.getInfos.infos
    #Notify the server that the task is done
    handle = False
    while not handle:
        try:
            client.taskDone("inverse", {"infos": infos})
            handle = True
        except BadStatusLine:
            print("[ERROR]  BadStatusLine - rescend the request")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("SYNTAX: python3.9 client.py config port_rpc")
        sys.exit(-1)

    config = ast.literal_eval(sys.argv[1])
    port_rpc = sys.argv[2]

    with xmlrpc.client.ServerProxy('http://localhost:'+ port_rpc) as client:
        handle = False
        while not handle:
            try:
                client.registerFirstObservation("inverse", [])
                handle = True
            except BadStatusLine:
                print("[ERROR]  BadStatusLine - rescend the request")

        root = init_simulation(config)

        command = None
        while command != "exit":
            handle = False
            while not handle:
                try:
                    task = client.getNextTask("inverse")
                    handle = True
                except BadStatusLine:
                    print("[ERROR]  BadStatusLine - rescend the request")

            command = task["command"]
            if command == "step":
                step(root, task["scale"], task["dt"], client)

            if command == "change_goal":
                pos_goal = task["new_pos"]
                root.moveGoal.update_goal(np.array(pos_goal))


    #Avoid that all client exit in the same time
    time.sleep(random.random())
    print("[INFO] >> Client is closed. Bye bye.")

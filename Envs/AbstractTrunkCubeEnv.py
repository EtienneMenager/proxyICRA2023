# -*- coding: utf-8 -*-
"""Specific environment for the gripper.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Feb 3 2021"

from sofagym.env.common.AbstractEnv import AbstractEnv
from sofagym.env.common.rpc_server import start_scene, get_position, get_infos, interact_scene
from sofagym.env.common.viewer import Viewer
from gym.envs.registration import register

import Sofa
import SofaRuntime
import importlib
import pygame
import glfw
import Sofa.SofaGL
from OpenGL.GL import *
from OpenGL.GLU import *

from gym import spaces
import os
import numpy as np

class AbstractTrunkCubeEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path =  os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "AbstractTrunkCube",
                      "deterministic": True,
                      "source": [150, 300, 500],
                       "target": [150, 30, 90],
                      "goalList": [[135, -12.9, 50]],
                      "cubePos": [100, -10, 45], 
                      "use_goal": False, 
                      "use_cube_pos": False,
                      "visualisation": False,
                      "render": 0,
                      "start_node": None,
                      "scale_factor": 30,
                      "dt": 0.01,
                      "timer_limit": 10,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "visuQP": True,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/AbstractTrunkCup",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": "python3.9",
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "use_abstract": True,
                      "inverse": False,
                      "use_her": False,
                      "max_move": -200,
                      "len_beam": 196.90922022411885,
                      "max_flex": [  0.011469505281464853,   0.049998633309957574],
                      "young_modulus": [ 14332.375222281547, 62717.82567449978],
                      "mass": [ 7.963338256107935,  4.805331266014146],
                      "rest_force": 60915.017308200295}

    def __init__(self, config = None):
        super().__init__(config)
        nb_actions = -1
        dim_action = 5
        low = np.array([-1]*dim_action)
        high = np.array([1]*dim_action)
        self.action_space = spaces.Box(low=low, high=high, shape=(dim_action,), dtype='float32')
        self.nb_actions = str(nb_actions)

        dim_state = 53
        low_coordinates = np.array([-500]*dim_state)
        high_coordinates = np.array([500]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')

    def step(self, action):
        return super().step(action)

    def reset(self, goal = None):
        super().reset()
        if not self.config["use_goal"]:
            self.goal = [float(np.random.uniform(60, 140)), -12.9, float(np.random.uniform(30, 70))]
        elif self.config["use_goal"]:
            self.goal = self.config["goalList"][0]
        else:
            self.goal = goal

        if not self.config["use_cube_pos"]:
            cubePos = self.config["cubePos"]
        else:
            cubePos = [float(np.random.uniform(80, 120)), -10, float(np.random.uniform(40, 50))]

        self.config.update({'goalPos': self.goal})
        self.config.update({'cubePos': cubePos})

        obs = start_scene(self.config, self.nb_actions)
        return np.array(obs['observation'])

    def get_available_actions(self):
        return self.action_space

    def render(self, mode='rgb_array'):
        if self.config['render']!=0 and self.config['visuQP']:
            #Define the viewer at the first run of render.
            if not self.viewer:
                display_size = self.config["display_size"]  # Sim display
                if 'zFar' in self.config:
                    zFar = self.config['zFar']
                else:
                    zFar = 0
                self.viewer = Viewer(self, display_size, zFar = zFar, save_path=self.config["save_path_image"])

            #Use the viewer to display the environment.
            self.viewer.render()


    def get_position(self):
        pos = get_position(self.past_actions)['position']
        if pos == []:
            return []
        poseTrunk, _, _, _, _, collision_1, collision_2 = pos[0][-2:]
        return poseTrunk, collision_1, collision_2

    
    def get_infos(self):
        infos = get_infos(self.past_actions)['infos']
        return infos

    def set_infos(self, infos):
        interact_scene(self.past_actions, infos)



register(
    id='abstracttrunkcube-v0',
    entry_point='sofagym.env:AbstractTrunkCubeEnv',
)

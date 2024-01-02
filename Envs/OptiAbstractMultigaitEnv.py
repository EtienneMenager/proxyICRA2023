# -*- coding: utf-8 -*-
"""Specific environment for the gripper.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Feb 3 2021"

from sofagym.env.common.AbstractEnv import AbstractEnv
from sofagym.env.common.rpc_server import start_scene, get_infos, interact_scene

from gym.envs.registration import register

from gym import spaces
import os
import numpy as np

class OptiAbstractMultigaitEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "OptiAbstractMultigait",
                      "deterministic": True,
                     "source": [150.0, -500, 150],
                     "target": [150, 0, 0],
                     "goalList": [[0, 0, 0]],
                      "start_node": None,
                      "scale_factor": 60,
                      "dt": 0.01,
                      "timer_limit": 12,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 1,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/AbstractMultigait",
                      "planning": False,
                      "discrete": True,
                      "start_from_history": None,
                      "python_version": "python3.9",
                      "zFar": 4000,
                      "time_before_start": 20,
                      "seed": None,

                      "young_modulus": [4099.850479814699, 1344.7672234828424],
                      "mass_density": [2.0011452536508282e-05, 1.887567875388044e-07],
                      "def_leg": 0.041649610722372096,
                      "def_cent": 0.02193426390567341,
                      }


    def __init__(self, config = None):
        super().__init__(config)
        # #Continous setting
        # nb_actions = -1
        # n_dim = 3 #3 if symetric, 5 if non symetric
        # low = np.array([-1]*n_dim)
        # high = np.array([1]*n_dim)
        # self.action_space = spaces.Box(low=low, high=high, shape=(n_dim,), dtype='float32')
        # self.nb_actions = str(nb_actions)

        #Discrete setting
        nb_actions = 6
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)


        dim_state = 60
        low_coordinates = np.array([-100]*dim_state)
        high_coordinates = np.array([100]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')

    def step(self, action):
        return super().step(action)

    def reset(self):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()

        self.config.update({'goalPos': self.goal})
        obs = start_scene(self.config, self.nb_actions)

        return np.array(obs['observation'])

    def get_available_actions(self):
        """Gives the actions available in the environment.

        Parameters:
        ----------
            None.

        Returns:
        -------
            list of the action available in the environment.
        """
        return self.action_space

    def get_infos(self):
        infos = get_infos(self.past_actions)['infos']
        return infos

    def set_infos(self, infos):
        interact_scene(self.past_actions, infos)


register(
    id='optiabstractmultigait-v0',
    entry_point='sofagym.env:OptiAbstractMultigaitEnv',
)

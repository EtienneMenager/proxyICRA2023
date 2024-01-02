# -*- coding: utf-8 -*-
"""Create the scene with the Abstraction of Jimmy.


Units: cm, kg, s.
"""
VISUALISATION = True

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"


import sys
import importlib
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from common.header import addHeader
from common.header import addVisu
from common.utils import addRigidObject

from OptiAbstractMultigait import OptiAbstractMultigait
from OptiAbstractMultigaitToolbox import  goalSetter, sceneModerator, applyAction, rewardShaper, History

def add_goal_node(root, position = [0, 0.0, 0.0], name = "Goal"):
    goal = root.addChild(name)
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject = True, showObjectScale = 5, showColor = [255, 0, 0, 255], drawMode = 1, position=position)
    return goal_mo

import itertools
import numpy as np
from pyquaternion import Quaternion


def createScene(rootNode, config = {"source": [0, -50, 0],
                                            "target": [0, 0, 0],
                                            "goalPos": [0, 0, 400],
                                            "seed": None,
                                            "num_scene": 0,
                                             "zFar":4000,
                                             "dt": 0.01,
                                             "scale_factor": 60,
                                             "time_before_start":20,
                                            #
                                            # "young_modulus": [4000, 1300],
                                            # "mass_density": [2.0354318346867232e-05, 1.76824833362052e-07],
                                            # "def_leg": 0.04210615834735613,
                                            # "def_cent": 0.02196200354545308,

                                            "young_modulus": [4099.850479814699, 1344.7672234828424],
                                                "mass_density": [2.0011452536508282e-05, 1.887567875388044e-07],
                                                "def_leg": 0.041649610722372096,
                                                "def_cent": 0.02193426390567341,
                                              },
                         mode = 'simu_and_visu'):

    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True


    addHeader(rootNode, alarmDistance=5, contactDistance=0.5, tolerance = 1e-6, maxIterations=100, gravity = [0,0,-9810], dt = config['dt'], mu='0.7')
    rootNode.addObject('RequiredPlugin', pluginName="SofaMiscMapping")

    position_spot = [[0, -100, 100], [0, 100, 100], [-100, 0, 100],
                     [100, 0, 100], [-100, 200, 150], [200, 200, 100]]
    direction_spot = [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
                      [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
    addVisu(rootNode, config, position_spot, direction_spot, cutoff = 250)


    multigait_config = {"tot_length_center": 59, "max_flex_center":config["def_cent"],
                        "tot_length_leg": 69, "max_flex_leg": config["def_leg"],
                         "init_pos": [0, 0, 0], "init_or": -5,
                         "young_modulus": config["young_modulus"], "mass_density": config["mass_density"],
                        }
    cosserat_config = {"nbSectionS_center":1, "nbFramesF_center":3,
                        "nbSectionS_leg":1, "nbFramesF_leg":3}


    optiAbstractMultigait = OptiAbstractMultigait(multigait_config = multigait_config, cosserat_config = cosserat_config)
    idx, same_direction, beams = optiAbstractMultigait.onEnd(rootNode)

    floor = addRigidObject(rootNode,filename='mesh/cube.obj',name='Floor',scale=[300, 1,200.0], position=[100.0,0,-20.0,  0.7071068, 0, 0, 0.7071068], collisionGroup=1)
    floor.addObject('FixedConstraint', indices=0)

    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode))
    rootNode.addObject(goalSetter(name="GoalSetter"))
    rootNode.addObject(sceneModerator(name="sceneModerator",  abstractMultigait = optiAbstractMultigait))
    rootNode.addObject(applyAction(name="applyAction", root= rootNode, abstractMultigait = optiAbstractMultigait, beams = beams, info_pos = [idx, same_direction]))
    # rootNode.addObject(History(name="History", rootNode=rootNode, use = True, waitingtime = config["time_before_start"], use_reward = True, use_points = True))

    if VISUALISATION:
        print(">> Add runSofa visualisation")
        from Controller import Visualisation

        # actions = [4, 0, 0, 2, 5, 1, 3, 4, 0, 0,
        #            2, 5, 1, 3, 4, 0, 0, 2]
        actions =  [4, 0, 2, 5, 1, 3]*3

        scale = config['scale_factor']
        time_before_start = config['time_before_start']
        rootNode.addObject(Visualisation(name="Visualisation", root = rootNode, actions = actions, scale = scale, time_before_start = time_before_start) )

    return rootNode

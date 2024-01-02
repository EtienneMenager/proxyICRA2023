# -*- coding: utf-8 -*-
"""Create the scene with the Abstraction of Jimmy.


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"


import sys
import importlib
import pathlib
import json
import os

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from common.header import addHeader
from common.header import addVisu
from common.utils import addRigidObject

from moveGoal import MoveGoal, MoveGoalMultigait

def createScene(rootNode, config = {"source": [50, -400, 150],
                                    "target": [50, 0, 0],
                                            "goalPos": [0, 0, 400],
                                            "seed": None,
                                            "num_scene": 0,
                                             "zFar":4000,
                                             "dt": 0.01,
                                             "scale_factor": 60
                                              },
                         mode = 'simu_and_visu'):

    addHeader(rootNode, alarmDistance=5, contactDistance=0.5, tolerance = 1e-6, maxIterations=100, gravity = [0,0,-9810], dt = config['dt'], mu='0.7')
    position_spot = [[0, -100, 100], [0, 100, 100], [-100, 0, 100],
                     [100, 0, 100], [-100, 200, 150], [200, 200, 100]]
    direction_spot = [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
                      [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
    addVisu(rootNode, config, position_spot, direction_spot, cutoff = 250)

    floor = addRigidObject(rootNode,filename='mesh/cube.obj',name='Floor',scale=[300, 1,200.0], position=[100.0,0,-20.0,  0.7071068, 0, 0, 0.7071068])
    floor.addObject('FixedConstraint', indices=0)

    # Goals
    with open("./TargetPoint.txt", 'r') as outfile:
        targetPointsPos = json.load(outfile)

    init_pos = targetPointsPos[0]

    goal = rootNode.addChild('Goals')
    GoalMO = goal.addObject('MechanicalObject', name='GoalMO', position=init_pos)
    goal.addObject('SphereCollisionModel', radius='1.0', group='0', color="green")

    rootNode.addObject(MoveGoal(name="MoveGoal", root = rootNode, positions = targetPointsPos))


    return rootNode

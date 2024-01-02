# -*- coding: utf-8 -*-
"""Create the scene with the Abstraction of Jimmy.


Units: cm, kg, s.
"""
VISUALISATION = False

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"


import sys
import importlib
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from common.header import addHeader as header
from common.header import addVisu as visu
from common.utils import addRigidObject

from AbstractMultigait import AbstractMultigait
from Controller import ControllerMultigait
from AbstractMultigaitToolbox import rewardShaper, goalSetter, sceneModerator, applyAction

def createScene(rootNode, config = {"source": [150.0, -500, 150],
                                    "target": [150, 0, 0],
                                            "goalPos": [0, 0, 0],
                                            "seed": None,
                                            "num_scene": 0,
                                             "zFar":4000,
                                             "dt": 0.01},
                         mode = 'simu_and_visu'):


    header(rootNode, alarmDistance=5, contactDistance=0.5, tolerance = 1e-6, maxIterations=100, gravity = [0, 0, -9810], dt = config['dt'], mu='0.7')

    position_spot = [[0, -100, 100], [0, 100, 100], [-100, 0, 100],
                     [100, 0, 100], [-100, 200, 150], [200, 200, 100]]
    direction_spot = [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
                      [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
    visu(rootNode, config, position_spot, direction_spot, cutoff = 250)


    multigait_config = {"tot_length_center": 42.92347544687069, "max_flex_center":0.03997484551000557,
                        "tot_length_leg":  59.30627065543256, "max_flex_leg": 0.03428811977536546,
                         "init_pos": [0, 0, 0]}
    cosserat_config = {"nbSectionS_center":1, "nbFramesF_center":2,
                        "nbSectionS_leg":1, "nbFramesF_leg":3}


    abstractMultigait = AbstractMultigait(multigait_config = multigait_config, cosserat_config = cosserat_config)
    abstractMultigait.onEnd(rootNode)

    multigait = abstractMultigait.multigait
    beams = [abstractMultigait.beam_center_1, abstractMultigait.beam_center_2,
            abstractMultigait.beam_legFR,
             abstractMultigait.beam_legFL, abstractMultigait.beam_legBR,
             abstractMultigait.beam_legBL]

    floor = addRigidObject(rootNode,filename='mesh/cube.obj',name='Floor',scale=[300.0,1,200.0], position=[100.0,0,-20.0,  0.7071068, 0, 0, 0.7071068], collisionGroup=1)
    floor.addObject('FixedConstraint', indices=0)

    # rootNode.addObject(ControllerMultigait(name="Controller", root= rootNode, multigait = multigait, beams = beams))

    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode))
    rootNode.addObject(goalSetter(name="GoalSetter"))
    rootNode.addObject(sceneModerator(name="sceneModerator",  abstractMultigait = abstractMultigait))
    rootNode.addObject(applyAction(name="applyAction", root= rootNode, abstractMultigait = abstractMultigait, beams = beams, factor = 1.3149757377520934))


    if VISUALISATION:
        print(">> Add runSofa visualisation")
        from common.visualisation import visualisationRunSofa
        actions = [[-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
                                    [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
                                    [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                                    [-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
                                    [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
                                    [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0,-1.0, -1.0]]*100
        scale = 60


        rootNode.addObject(visualisationRunSofa(name="visualisationRunSofa", root = rootNode, actions = actions, scale = scale) )

    # rootNode.init()
    # rootNode.Reward.update()

    return rootNode

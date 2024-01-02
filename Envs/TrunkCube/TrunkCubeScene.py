import Sofa
import math

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from math import cos
from math import sin
from pyquaternion import Quaternion
import numpy as np

from common.header import addHeader as header
from common.header import addVisu as visu
from common.utils import addRigidObject
from TrunkCube import Cube, Trunk
from Controller import ControllerTrunkCube, ControllerTrunkCubeTotal
from TrunkCubeToolbox import rewardShaper, goalSetter, sceneModerator, applyAction, Visualisation

import os
import json
path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
dirPath = os.path.dirname(os.path.abspath(__file__))+'/'
meshPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
pathSceneFile = os.path.dirname(os.path.abspath(__file__))


def add_goal_node(root, pos):
    goal = root.addChild("Goal")

    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3, showColor=[1, 0, 0, 0.25], position=pos)

    goal_z = goal.addChild("Goal_z")
    goal_z.addObject('MeshOBJLoader', name = "loader", filename='mesh/cylinder.obj', scale3d=[1, 40, 1], rotation = [0, 0, 90], translation = [pos[0]+200, pos[1], pos[2]])
    goal_z.addObject('OglModel', src='@loader',color=[1, 0, 0, 0.25])

    goal_x = goal.addChild("Goal_x")
    goal_x.addObject('MeshOBJLoader', name = "loader", filename='mesh/cylinder.obj', scale3d=[1, 40, 1], rotation = [90, 0, 0], translation = [pos[0], pos[1], pos[2]-200])
    goal_x.addObject('OglModel', src='@loader',color=[1, 0, 0, 0.25])



def createScene(rootNode, config={"source": [150, 300, 500],
 								  "target": [150, 30, 90],
								  "goalPos": [132, -12.9, 50],
                                  "cubePos": [100, -10, 45],
								  "seed": None,
								  "zFar":4000,
								  "dt": 0.01,
                                  "scale_factor": 30,
                                  "idx":0,
                                  "readTime": 0.0,
                                  "waitingtime": 0,
                                  "max_move": -200},
								  mode = 'simu_and_visu'):

	rootNode.addObject('RequiredPlugin', name='SofaExporter')


    # #Push goal [132, -12.9, 50]
	# config["goalPos"] = [132, -12.9, 50]
	# specific_actions = np.array([[-1., -1.,  1.,  1.,  1.],
    #            [-1., -1., 1., 0.7668063, -1.],
    #            [-1., -1.,  1.,  1., -1.]])


	header(rootNode, alarmDistance=20.0, contactDistance=2, tolerance = 1e-6, maxIterations=100, gravity = [0,-9810,0], dt = config['dt'], mu = "0.9")


	position_spot = [[0, -50, 10]]
	direction_spot = [[0.0, 1, 0]]
	visu(rootNode, config, position_spot, direction_spot, cutoff = 250)

	cube_config = {"init_pos": config["cubePos"], "scale": [20, 40, 20], "density": 3.3e-8} #5e-7

	floor = addRigidObject(rootNode,filename=meshPath+'cube.obj',name='Floor',scale=[250.0,0.1,220.0], position=[20,-55,30,0,0,0,1], collisionGroup = 0)
	floor.addObject('FixedConstraint', indices=0)

	cube = Cube(cube_config=cube_config)
	cube.onEnd(rootNode, collisionGroup=2)
	# cube = None

	add_goal_node(rootNode, config["goalPos"])
	rootNode.addObject(goalSetter(name="GoalSetter", goalPos=config["goalPos"]))
	rootNode.addObject(rewardShaper(name="Reward", root = rootNode, goalPos=config["goalPos"]))

	trunk = Trunk(inverse = False)
	trunk.onEnd(rootNode, collisionGroup = 1)

	init_pos = np.array([[i*196.90922022411885/30, 0, 0, 0, 0, 0, 1] for i in range(30+1)])
	points =  list(range(0, 30, 2))

	effectors = trunk.trunk.addChild('Effectors')
	effectors.addObject('MechanicalObject', position = init_pos[points, :3], showObject = True, showObjectScale=10, showColor=[0, 1, 0, 1])
	effectors.addObject('BarycentricMapping')

	actuators = trunk.cables + [trunk.sliding]

	rootNode.addObject(sceneModerator(name="sceneModerator",  root = rootNode, cube = cube, trunk = trunk, effectors = effectors, actuators = actuators))
	rootNode.addObject(applyAction(name="applyAction", root= rootNode))
	rootNode.addObject(Visualisation(name="Visu", root = rootNode))

	return rootNode

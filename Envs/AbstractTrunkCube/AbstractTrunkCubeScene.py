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
from common.controller_with_proxy import ControllerWithProxy, ControllerInverseWithProxy
from AbstractTrunkCube import AbstractTrunk, Cube, Trunk
from Controller import ControllerTrunkCube, ControllerTrunkCubeTotal
from AbstractTrunkCubeToolbox import rewardShaper, goalSetter, sceneModerator, applyAction, History

import os
import json
path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
dirPath = os.path.dirname(os.path.abspath(__file__))+'/'
meshPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
pathSceneFile = os.path.dirname(os.path.abspath(__file__))

class GetInfos(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.effectors = kwargs["effectors"]
        self.actuators = kwargs["actuators"]
        self.infos = None

    def _getInfos(self):
        actuation = [float(actuator.displacement.value) for actuator in self.actuators[:-1]] + [self.actuators[-1].position.value[:].tolist()]
        effectorsPos = self.effectors.MechanicalObject.position.value[:].tolist()
        self.infos = {"actuation": actuation, "effectorsPos": effectorsPos, "points": []}

    def getInfos(self):
        barycentre = self.effectors.MechanicalObject.position.value[:,:3].mean(axis = 0)
        cube_pos = self.root.Cube.MechanicalObject.position.value
        return {"barycentre" :barycentre.tolist(), "cube_pos": cube_pos.tolist()}


    def onAnimateEndEvent(self, event):
        self._getInfos()

class GetReward(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.goal_pos =  kwargs["goal_pos"]

    def update(self):
        pass

    def getReward(self):
        pos_cube = self.root.Cube.MechanicalObject.position.value[0, :3]
        current_dist_cube_goal = self._compute_dist(pos_cube, self.goal_pos)
        return current_dist_cube_goal, current_dist_cube_goal<8

    def _compute_dist(self, pos, goal):
            A,B = np.array(pos)[[0, 2]], np.array(goal)[[0, 2]]
            return float(np.linalg.norm(A-B))

class Actuate(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.actuators = kwargs["actuators"]
        self.effectors = kwargs["effectors"]
        self.values = None

        self.current_time = 0

    def getInfos(self):
        barycentre = self.effectors.MechanicalObject.position.value[:,:3].mean(axis = 0)
        cube_pos = self.root.Cube.MechanicalObject.position.value[0]
        return {"barycentre" :barycentre.tolist(), "cube_pos": cube_pos.tolist()}

    def _setValue(self, values):
        self.values = values

    def save_actuators_state(self):
        forces = []
        for actuator in self.actuators[:-1]:
            forces.append(actuator.force.value)
        forces.append(self.actuators[-1].position.value[:].tolist())
        np.savetxt(pathSceneFile + "/CablesForces.txt", forces)
        print("[INFO]  >>  Save pressure at "+pathSceneFile + "/CablesForces.txt")

    def onAnimateBeginEvent(self, event):
        if not(self.values is None):
            for actuator, value in zip(self.actuators[:-1], self.values[:-1]):
                actuator.value.value = [value]
            self.actuators[-1].rest_position.value = self.values[-1]

    def action_rescaling(self, points_sofagym, points_inverse):
        return 1


class MoveGoal(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.goals = kwargs["goals"]
        self.inverse_model = kwargs["inverse_model"]
        self.waitingtime = kwargs["waitingtime"]
        self.inverse = kwargs["inverse"]
        self.effectors = kwargs["effectors"]


        self.goal_pos = None
        self.idx_goal = kwargs["idx_goal"]
        self.particular_points = kwargs["particular_points"]

        self.idx_pos = 0
        self.pos = None

    def update_goal(self, goal_pos):
        self.goal_pos = goal_pos
        self.idx_goal = 0

    def update_pos(self, pos):
        self.pos = pos
        self.idx_pos = 0

    def getCorrectedPID(self, position, Kp):
        effectorsPos_direct = self.effectors.MechanicalObject.position.value[:, :3]
        return position[0][self.particular_points] + Kp*(position[0][self.particular_points]-effectorsPos_direct)

    def onAnimateBeginEvent(self, event):
        # if not self.inverse:
        #     print("Effectors:",  self.root.solverNode.reducedModel.model.Effectors.MechanicalObject.position.value[:, :3].mean(axis = 0))

        if not (self.goal_pos is None) and self.idx_goal < len(self.goal_pos):
            if self.inverse:
                # #if in controler direct
                # pos = self.goal_pos
                # self.goals[0].PositionEffector.effectorGoal.value = pos

                #if in controler invers
                pos = self.goal_pos[self.idx_goal][self.particular_points]
                current_pos = self.goals[0].MechanicalObject.position.value[:]
                corrected_pos = pos + 0.3*(pos - current_pos)

                self.goals[0].PositionEffector.effectorGoal.value = corrected_pos
                self.goals[1].position.value = pos
            else:
                pos = self.goal_pos[self.idx_goal][self.particular_points]
                self.goals.position.value = pos
            self.idx_goal+=1

        if not (self.pos is None) and self.idx_pos < len(self.pos) and not self.inverse:
            pos = self.pos[self.idx_pos]
            self.inverse_model.position.value = pos
            self.idx_pos+=1


def add_goal_node(root, pos):
    goal = root.addChild("Goal")

    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3, showColor=[1, 0, 0, 0.25], position=pos)

    goal_z = goal.addChild("Goal_z")
    goal_z.addObject('MeshOBJLoader', name = "loader", filename='mesh/cylinder.obj', scale3d=[1, 40, 1], rotation = [0, 0, 90], translation = [pos[0]+200, pos[1], pos[2]])
    goal_z.addObject('OglModel', src='@loader',color=[1, 0, 0, 0.25])

    goal_x = goal.addChild("Goal_x")
    goal_x.addObject('MeshOBJLoader', name = "loader", filename='mesh/cylinder.obj', scale3d=[1, 40, 1], rotation = [90, 0, 0], translation = [pos[0], pos[1], pos[2]-200])
    goal_x.addObject('OglModel', src='@loader',color=[1, 0, 0, 0.25])


def create_rigide_from_beam(parent, beam, points_in_beams, angles, filename = "./pos_end_beams.txt"):
    bendingEffectors = parent.addChild("bendingEffectors")
    point_sort, argpoint_sort = np.sort(points_in_beams), np.argsort(points_in_beams)
    pos = beam.position.value[point_sort, :3].tolist()
    for i in range(len(pos)):
        quat = Quaternion(axis=[1, 0, 0], degrees=angles[argpoint_sort[i]])
        pos[i] = pos[i] + list(quat)
    bendingEffectors.addObject("MechanicalObject", name="MO", template = "Rigid3d", position = pos, showObject = True, showObjectScale=10, showColor=[0, 0, 1, 1])

    repartition = [0 for _ in range(beam.position.value.shape[0])]
    for id in points_in_beams:
        repartition[id]+=1
    bendingEffectors.addObject("RigidRigidMapping", repartition = repartition, mapForces=True, mapMasses=True, globalToLocalCoords = True)
    # with open(filename, 'w') as fp:
    #     json.dump(pos, fp)


def create_rigide_from_file(parent,  translation = [0, 0, 0], filename = "./pos_end_beams.txt"):
    with open(filename, 'r') as outfile:
        pos = np.array(json.load(outfile))
    pos[:,0]+= translation[0]
    pos[:,1]+= translation[1]
    pos[:,2]+= translation[2]

    bendingEffectors = parent.addChild("bendingEffectors")
    bendingEffectors.addObject("MechanicalObject", name="MO", template = "Rigid3d", position = pos, showObject = True, showObjectScale=10, showColor=[0, 0, 1, 1])
    bendingEffectors.addObject("BarycentricMapping")
    # bendingEffectors.addObject('PositionEffector', template='Rigid3d', useDirections= [0, 0, 0, 1, 1, 1], effectorGoal= init_pos, indices = [i for i in range(len(init_pos))])



def createScene(rootNode, config={"source": [150, 300, 500],
 								  "target": [150, 30, 90],
								  "goalPos": [132, -12.9, 50],
                                  "cubePos": [100, -10, 45],
								  "seed": None,
								  "zFar":4000,
								  "dt": 0.01,
                                  "scale_factor": 30,
								  "use_abstract": False,
                                  "inverse": True,
                                  "visualisation": False,
                                  "idx":0,
                                  "readTime": 0.0,
                                  "waitingtime": 0,
                                  "max_move": -200,
                                  "len_beam": 196.90922022411885,
                                  "max_flex": [  0.011469505281464853,   0.049998633309957574],
                                  "young_modulus": [ 14332.375222281547, 62717.82567449978],
                                  "mass": [ 7.963338256107935,  4.805331266014146],
                                  "rest_force": 60915.017308200295},
								  mode = 'simu_and_visu'):

	rootNode.addObject('RequiredPlugin', name='SofaExporter')


    # #Push goal [132, -12.9, 50]
	config["goalPos"] = [132, -12.9, 50]
	specific_actions = np.array([[-1., -1.,  1.,  1.,  1.],
               [-1., -1., 1., 0.7668063, -1.],
               [-1., -1.,  1.,  1., -1.]])


	if config["use_abstract"]:
		header(rootNode, alarmDistance=20.0, contactDistance=2, tolerance = 1e-6, maxIterations=100, gravity = [0,-9810,0], dt = config['dt'], mu = "0.9")
	elif not config["inverse"]:
		header(rootNode, alarmDistance=20.0, contactDistance=2, tolerance = 1e-6, maxIterations=100, gravity = [0,-9810,0], dt = config['dt'], mu = "0.9")
	else:
		header(rootNode, alarmDistance=20.0, contactDistance=3, tolerance = 1e-6, maxIterations=100, gravity = [0,-9810,0], dt = config['dt'], mu = "0.9", genericConstraintSolver = False)
		rootNode.addObject('RequiredPlugin', name='SoftRobots.Inverse')
		rootNode.addObject("QPInverseProblemSolver", allowSliding=True, epsilon = 0.045, responseFriction = 0.9)


	position_spot = [[0, -50, 10]]
	direction_spot = [[0.0, 1, 0]]
	visu(rootNode, config, position_spot, direction_spot, cutoff = 250)

	USE_NOISE = True
	level_noise = 0.1
	if USE_NOISE:

		config["len_beam"] = (1+level_noise*(2*np.random.random()-1))*config["len_beam"]
		config["rest_force"] = (1+level_noise*(2*np.random.random()-1))*config["rest_force"]
		config["max_flex"][0] = (1+level_noise*(2*np.random.random()-1))*config["max_flex"][0]
		config["max_flex"][1] = (1+level_noise*(2*np.random.random()-1))*config["max_flex"][1]
		config["young_modulus"][0] = (1+level_noise*(2*np.random.random()-1))*config["young_modulus"][0]
		config["young_modulus"][1] = (1+level_noise*(2*np.random.random()-1))*config["young_modulus"][1]
		config["mass"][0] = (1+level_noise*(2*np.random.random()-1))*config["mass"][0]
		config["mass"][1] = (1+level_noise*(2*np.random.random()-1))*config["mass"][1]

	cosserat_config = {'init_pos': [0, 0, 0], 'tot_length': config["len_beam"], 'nbSectionS': 6, 'nbFramesF': 30}
	trunk_config = {"max_flex": config["max_flex"], "max_move": config["max_move"], "max_angular_rate": max(config["max_flex"]), "max_incr": 5, "base_size": [2, 25, 25],
                    "young_modulus": config["young_modulus"], "mass": config["mass"]}

	if config["use_abstract"]:
		cube_config = {"init_pos": config["cubePos"], "scale": [20, 40, 20], "density": 2.3e-5}
	else:
		cube_config = {"init_pos": config["cubePos"], "scale": [20, 40, 20], "density": 3.3e-8} #5e-7

	floor = addRigidObject(rootNode,filename=meshPath+'cube.obj',name='Floor',scale=[250.0,0.1,220.0], position=[20,-55,30,0,0,0,1], collisionGroup = 0)
	floor.addObject('FixedConstraint', indices=0)

	cube = Cube(cube_config=cube_config)
	cube.onEnd(rootNode, collisionGroup=2)
	# cube = None

	add_goal_node(rootNode, config["goalPos"])
	rootNode.addObject(goalSetter(name="GoalSetter", goalPos=config["goalPos"]))
	rootNode.addObject(rewardShaper(name="Reward", root = rootNode, goalPos=config["goalPos"]))

	if config["use_abstract"]:
		abstractTrunk = AbstractTrunk(cosserat_config = cosserat_config, trunk_config=trunk_config)
		abstractTrunk.onEnd(rootNode, collisionGroup=1)
		# rootNode.addObject(ControllerTrunkCube(name="Controller", root = rootNode, trunk = abstractTrunk))

		abstractTrunk.trunk.addObject("RestShapeSpringsForceField", points = [i for i in range(31)], stiffness = config["rest_force"], angularStiffness = 1e8) #config["rest_force"])
		rootNode.addObject(sceneModerator(name="sceneModerator",  cube = cube, trunk = abstractTrunk))
		rootNode.addObject(applyAction(name="applyAction", root= rootNode, trunk=abstractTrunk))

		create_rigide_from_beam(abstractTrunk.trunk, abstractTrunk.trunk.MechanicalObject, [1, 15, cosserat_config["nbFramesF"]], [0, 0, 0], filename = "./pos_end_beams.txt")

        # Needed for transfert
		rootNode.addObject(History(name="History", rootNode=rootNode, use = True, use_reward = True, waitingtime = 0))

	else:
		trunk = Trunk(trunk_config=trunk_config, inverse = config["inverse"])
		trunk.onEnd(rootNode, collisionGroup = 1)
		# rootNode.addObject(ControllerTrunkCubeTotal(name="Controller", root = rootNode))

		init_pos = np.array([[i*cosserat_config['tot_length']/cosserat_config['nbFramesF'], 0, 0, 0, 0, 0, 1] for i in range(cosserat_config['nbFramesF']+1)])
		points =  list(range(0, 30, 2))#list(range(2))+list(range(24,30))# list(range(30)) # list(range(20, 30))

		if config["inverse"]:
			actuators = trunk.cables + [trunk.sliding]

			effectors = trunk.trunk.addChild('Effectors')
			effectors.addObject('MechanicalObject', position = init_pos[points, :3], showObject = True, showObjectScale=10, showColor=[1, 0, 0, 1])
			effectors.addObject('PositionEffector', effectorGoal= init_pos[points, :3], indices = [i for i in range(len(points))], weight = 1)
			effectors.addObject('BarycentricMapping')

			goals = rootNode.addChild("Goals")
			goals.addObject('MechanicalObject', position=init_pos[points, :3], showObject = True, showObjectScale=10, showColor=[0, 0, 1, 1])


			rootNode.addObject(GetInfos(name="getInfos", root = rootNode, actuators = actuators, effectors = effectors))
			rootNode.addObject(MoveGoal(name="moveGoal",root = rootNode,  goals = [effectors, goals.MechanicalObject], waitingtime = config["waitingtime"], idx_goal = config["idx"], inverse= True, inverse_model = None, effectors = None, particular_points = points))

			rootNode.addObject(ControllerInverseWithProxy(name="ControllerInverseWithProxy", root = rootNode,
                                scale = 30, env_name = "abstracttrunkcube-v0", save_rl_path = "../../../Results_benchmark/PPO_abstracttrunkcube-v0_10/best",
                                name_algo = "PPO", config = config,  translation = np.array([0, 0, 0]),
                                time_register_sofagym = 1000, specific_actions = specific_actions, name_save = "../../../Proxy/Results/TrunkCube_Push_"+str(level_noise)+"/",
                                effectors = effectors, goals = goals, use_cumulative_reward_sofagym = False))
			rootNode.addObject(GetReward(name="getReward", root = rootNode, goal_pos = config["goalPos"]))

			# try:
			# 	readTime = config["readTime"]
			# 	trunk.trunk.addObject('ReadState', name="states", filename= pathSceneFile+"/trunkState", shift = readTime, printLog = "1")
			# 	trunk.trunk.collision1.addObject('ReadState', name="states", filename= pathSceneFile+"/collision1State", shift = readTime, printLog = "1")
			# 	trunk.trunk.collision2.addObject('ReadState', name="states", filename= pathSceneFile+"/collision2State", shift = readTime, printLog = "1")

			# 	trunk.trunk.cableL0.addObject('ReadState', name="states", filename= pathSceneFile+"/cableL0State", shift = readTime, printLog = "1")
			# 	trunk.trunk.cableL1.addObject('ReadState', name="states", filename= pathSceneFile+"/cableL1State", shift = readTime, printLog = "1")
			# 	trunk.trunk.cableL2.addObject('ReadState', name="states", filename= pathSceneFile+"/cableL2State", shift = readTime, printLog = "1")
			# 	trunk.trunk.cableL3.addObject('ReadState', name="states", filename= pathSceneFile+"/cableL3State", shift = readTime, printLog = "1")

			# 	trunk.trunk.cableS0.addObject('ReadState', name="states", filename= pathSceneFile+"/cableS0State", shift = readTime, printLog = "1")
			# 	trunk.trunk.cableS1.addObject('ReadState', name="states", filename= pathSceneFile+"/cableS1State", shift = readTime, printLog = "1")
			# 	trunk.trunk.cableS2.addObject('ReadState', name="states", filename= pathSceneFile+"/cableS2State", shift = readTime, printLog = "1")
			# 	trunk.trunk.cableS3.addObject('ReadState', name="states", filename= pathSceneFile+"/cableS3State", shift = readTime, printLog = "1")

			# except:
			# 	print("[WARNING]  >> No read state available. Make sure filenames exist.")


		else:
			writeTime = 30*0.01
			scale = 0

			w_1 = trunk.trunk.addObject('WriteState', name="writer", filename= pathSceneFile+"/trunkState",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)
			w_2 = trunk.trunk.collision1.addObject('WriteState', name="writer", filename= pathSceneFile+"/collision1State",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)
			w_3 = trunk.trunk.collision2.addObject('WriteState', name="states", filename= pathSceneFile+"/collision2State",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)

			w_4 = trunk.trunk.cableL0.addObject('WriteState', name="writer", filename= pathSceneFile+"/cableL0State",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)
			w_5 = trunk.trunk.cableL1.addObject('WriteState', name="writer", filename= pathSceneFile+"/cableL1State",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)
			w_6 = trunk.trunk.cableL2.addObject('WriteState', name="writer", filename= pathSceneFile+"/cableL2State",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)
			w_7 = trunk.trunk.cableL3.addObject('WriteState', name="writer", filename= pathSceneFile+"/cableL3State",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)

			w_8 = trunk.trunk.cableS0.addObject('WriteState', name="writer", filename= pathSceneFile+"/cableS0State",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)
			w_9 = trunk.trunk.cableS1.addObject('WriteState', name="writer", filename= pathSceneFile+"/cableS1State", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)
			w_10 = trunk.trunk.cableS2.addObject('WriteState', name="writer", filename= pathSceneFile+"/cableS2State",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)
			w_11= trunk.trunk.cableS3.addObject('WriteState', name="writer", filename= pathSceneFile+"/cableS3State",time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)
			writers = [w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11]


			effectors = trunk.trunk.addChild('Effectors')
			effectors.addObject('MechanicalObject', position = init_pos[points, :3], showObject = True, showObjectScale=10, showColor=[0, 1, 0, 1])
			effectors.addObject('BarycentricMapping')

			goals = rootNode.addChild("Goals")
			goals.addObject('MechanicalObject', position=init_pos[points, :3], showObject = True, showObjectScale=10, showColor=[0, 0, 1, 1])

			inverse_model = rootNode.addChild("inverseModel")
			inverse_model.addObject('MechanicalObject', position=init_pos[points, :3], showObject = True, showObjectScale=5, showColor=[1, 0, 0, 1])

			actuators = trunk.cables + [trunk.sliding]

            #Controller with proxy
			config.update({"inverse": True})
			noise_level = 0
			rootNode.addObject(ControllerWithProxy(name="ControllerWithProxy", root = rootNode, inverse_scene_path = "sofagym.env.AbstractTrunkCube.AbstractTrunkCubeScene",
                                scale = 30, env_name = "abstracttrunkcube-v0", save_rl_path = "../../../Results_benchmark/PPO_abstracttrunkcube-v0_40/best",
                                name_algo = "PPO", config = config, writers = writers, translation = np.array([0, 0, 0]),
                                time_register_sofagym = 10, time_register_inverse = 10, nb_step_inverse=3, init_action = [0, 0, 0, 0, 0, 0, 0, 0],
                                min_bounds = [0, 0, 0, 0, 0, 0, 0, 0], max_bounds = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
                                factor_commande = 1, Kp = 0, specific_actions = specific_actions, name_save = "../../../Proxy/Results/TrunkCube_"+str(level_noise)+"/",
                                effectors = effectors, goals = goals, use_cumulative_reward_sofagym = False))



			rootNode.addObject(GetReward(name="getReward", root = rootNode, goal_pos = config["goalPos"]))
			rootNode.addObject(Actuate(name="actuate", effectors = effectors, actuators = actuators, root = rootNode, translation = np.array([0, 0, 0])))
			rootNode.addObject(MoveGoal(name="moveGoal", root = rootNode, effectors = effectors, goals = goals.MechanicalObject, waitingtime = config["waitingtime"], idx_goal = config["idx"], inverse= False, translation =np.array([0, 0, 0]), inverse_model = inverse_model.MechanicalObject, particular_points = points))




	if config["visualisation"] and config["use_abstract"]:
		print(">> Add runSofa visualisation")
		from Controller import Visualisation

        #goal [-65, -12.9, 90]
		# actions = [[-1., -1., -1., -1., -1.],
        #            [1., 1., 1., 1., 1.],
        #            [1., -1.,  1., -1.,  1.],
        #            [-1.,  1., -1., -1., -1.],
        #            [-1.,  1., -0.1137368,  -1., -1.],
        #            [1., 1., 1., 1., 1.],
        #            [1., -0.3834106,  1., -1., 1.]]

        # #goal [-42, -12.9, 45]
		# actions = [[-1., -1., -1., -1., -1.],
        #            [1., 1., 1., 1., 1.]]

        #goal [132, -12.9, 50]
		actions = [[-1., -1.,  1.,  1.,  1.],
                   [-1., -1., 1., 0.7668063, -1.],
                   [-1., -1.,  1.,  1., -1.]]

        #Bayesian optimisation
		# actions = [[0, -1, 0, 0, -1],
        #            [0, 0, 0, -0.5, -1],
        #            [0, 1, 0, -1, -1]]

		scale = config['scale_factor']
		rootNode.addObject(Visualisation(name="Visualisation", root = rootNode, actions = actions, scale = scale) )


	return rootNode

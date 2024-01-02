# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Fab 3 2021"

import numpy as np
from pyquaternion import Quaternion
import json
import os

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


SofaRuntime.importPlugin("SofaComponentAll")

class rewardShaper(Sofa.Core.Controller):
        """Compute the reward.

        Methods:
        -------
            __init__: Initialization of all arguments.
            getReward: Compute the reward.
            update: Initialize the value of cost.

        Arguments:
        ---------
            rootNode: <Sofa.Core>
                The scene.
            goal_pos: coordinates
                The position of the goal.
            effMO: <MechanicalObject>
                The mechanical object of the element to move.
            cost:
                Evolution of the distance between object and goal.

        """
        def __init__(self, *args, **kwargs):
            """Initialization of all arguments.

            Parameters:
            ----------
                kwargs: Dictionary
                    Initialization of the arguments.

            Returns:
            -------
                None.

            """
            Sofa.Core.Controller.__init__(self, *args, **kwargs)

            self.root = kwargs["root"]
            self.goal_pos = np.array(kwargs["goalPos"])

        def getReward(self):
            pos_cube = self.root.sceneModerator.cube.cube.MechanicalObject.position.value[0, :3]
            current_dist_cube_goal = self._compute_dist(pos_cube, self.goal_pos)

            #For learning
            pos_trunk = self.root.sceneModerator.effectors.MechanicalObject.position.value[:, :3]
            pos_colliCube = self.root.sceneModerator.cube.cube.Collision.MechanicalObject.position.value[:, :3]
            face_pos = [pos_colliCube[[0, 1, 4, 5]].mean(axis=0), pos_colliCube[[1, 2, 5, 6]].mean(axis=0), pos_colliCube[[2, 3, 6, 7]].mean(axis=0), pos_colliCube[[3, 0, 7, 4]].mean(axis=0)]
            dist_face = [self._compute_dist(self.goal_pos, p) for p in face_pos]
            id = np.argmax(dist_face)

            opposite_face = np.array(face_pos[id])
            current_dist_opposite_face_tips = float(np.min(np.linalg.norm(pos_trunk-opposite_face, axis = 1)))

            factor = [0.7, 0.3]
            dist_factor = 50
            reward = factor[0]*(self.init_dist - current_dist_cube_goal)/self.init_dist + factor[1]*min(1, dist_factor/current_dist_opposite_face_tips)

            return reward, current_dist_cube_goal

        def update(self):
            pos_cube = self.root.sceneModerator.cube.cube.MechanicalObject.position.value[0, :3]
            self.init_dist = self._compute_dist(pos_cube, self.goal_pos)

        def _compute_dist(self, pos, goal):
            A,B = pos[[0, 2]], goal[[0, 2]]
            # print(A, B)
            return float(np.linalg.norm(A-B))


class goalSetter(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.goalPos = kwargs["goalPos"]

    def getGoal(self, factor = None):
        return [self.goalPos[0], self.goalPos[2]]

    def update(self):
        pass

    def set_mo_pos(self, goal):
        pass


class sceneModerator(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        self.cube = kwargs["cube"]
        self.trunk = kwargs["trunk"]
        self.effectors = kwargs["effectors"]
        self.actuators = kwargs["actuators"]

    def getPos(self):
        pos_cube = self.cube.getPos()
        pos_trunk = self.trunk.getPos()
        return [pos_cube, pos_trunk]

    def setPos(self, pos):
        self.cube.setPos(pos[0])
        self.trunk.setPos(pos[1])



################################################################################

def getState(rootNode):
    effectors = rootNode.sceneModerator.effectors
    cube = rootNode.sceneModerator.cube

    effectorsPos = effectors.MechanicalObject.position.value[:, :3].reshape(-1).tolist()
    cubPos = cube.cube.MechanicalObject.position.value[0, :3].tolist()
    goal = rootNode.GoalSetter.getGoal()

    state = effectorsPos + cubPos + goal
    return state


def getReward(rootNode, bounds = [5, 400], penality = 0): #NOTE: penality = 10 for the training
    reward, dist =  rootNode.Reward.getReward()

    reward = dist
    done = True
    if dist > bounds[1]:
        r = reward - penality
    elif dist < bounds[0]:
        r = reward + penality
    else:
        r = reward
        done = False

    return done, r


def getPos(root):
    position = root.sceneModerator.getPos()
    return position

def setPos(root, pos):
    root.sceneModerator.setPos(pos)

################################################################################

class applyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]

        self.max_force_1 = 30
        self.max_force_2 = 15
        self.max_move = -200
        self.max_incr_force = 1
        self.max_incr_move = 5

        self.a_move, self.b_move =  self.max_move/2 , self.max_move/2
        self.a_pull_1, self.b_pull_1 = self.max_force_1/2, self.max_force_1/2
        self.a_pull_2, self.b_pull_2 = self.max_force_2/2, self.max_force_2/2

        # self.step = 0
        print(">>  Init done.")

    def _normalizedAction_to_action(self, action, type, num_section):
        if type == "cable":
            if num_section == 1:
                return self.a_pull_1*action + self.b_pull_1
            else:
                return self.a_pull_2*action + self.b_pull_2
        else:
            return self.a_move*action + self.b_move

    def _controlIncr(self, incr, type):
        max_incr = self.max_incr_force if type == "cable" else self.max_incr_move

        if abs(incr)>max_incr:
            if incr>=0:
                incr = max_incr
            else:
                incr = -max_incr
        return incr

    def _moveBase(self, incr):
        actuator = self.root.sceneModerator.trunk.sliding
        if actuator.rest_position.value[0,0] + incr >= self.max_move and actuator.rest_position.value[0,0] + incr <=0:
            with actuator.rest_position.writeable() as pos:
                pos[:,0]+=incr

        else:
            if actuator.rest_position.value[0,0] + incr < self.max_move:
                with actuator.rest_position.writeable() as pos:
                    pos[:,0]=self.max_move
            else:
                with actuator.rest_position.writeable() as pos:
                    pos[:,0]=0

    def _pull(self, numCable, incr):
        cable = self.root.sceneModerator.actuators[numCable]
        if numCable in [0, 1, 2, 3]:
            max_force = self.max_force_1
        else:
            max_force = self.max_force_2

        if abs(cable.value[0] + incr)<= max_force:
            cable.value = [cable.value[0] + incr]
        else:
            if cable.value[0] + incr < 0:
                cable.value = [0]
            else:
                cable.value = [max_force]

    def compute_action(self, actions, nb_step):
        goal_cable_1 = self._normalizedAction_to_action(actions[0], type = "cable", num_section=1)
        goal_cable_2 = self._normalizedAction_to_action(actions[1], type = "cable", num_section=1)
        goal_cable_3 = self._normalizedAction_to_action(actions[2], type = "cable", num_section=1)
        goal_cable_4 = self._normalizedAction_to_action(actions[3], type = "cable", num_section=1)
        goal_cable_5 = self._normalizedAction_to_action(actions[4], type = "cable", num_section=2)
        goal_cable_6 = self._normalizedAction_to_action(actions[5], type = "cable", num_section=2)
        goal_cable_7 = self._normalizedAction_to_action(actions[6], type = "cable", num_section=2)
        goal_cable_8 = self._normalizedAction_to_action(actions[7], type = "cable", num_section=2)
        goal_base_pos = self._normalizedAction_to_action(actions[8], type = "base", num_section=None)

        current_cable_1 = self.root.sceneModerator.actuators[0].value[0]
        current_cable_2 = self.root.sceneModerator.actuators[1].value[0]
        current_cable_3 = self.root.sceneModerator.actuators[2].value[0]
        current_cable_4 = self.root.sceneModerator.actuators[3].value[0]
        current_cable_5 = self.root.sceneModerator.actuators[4].value[0]
        current_cable_6 = self.root.sceneModerator.actuators[5].value[0]
        current_cable_7 = self.root.sceneModerator.actuators[6].value[0]
        current_cable_8 = self.root.sceneModerator.actuators[7].value[0]
        current_base_pos = self.root.sceneModerator.actuators[8].rest_position.value[0][0]

        incr_cable_1 = self._controlIncr((goal_cable_1 - current_cable_1)/nb_step, type = "cable")
        incr_cable_2 = self._controlIncr((goal_cable_2 - current_cable_2)/nb_step, type = "cable")
        incr_cable_3 = self._controlIncr((goal_cable_3 - current_cable_3)/nb_step, type = "cable")
        incr_cable_4 = self._controlIncr((goal_cable_4 - current_cable_4)/nb_step, type = "cable")
        incr_cable_5 = self._controlIncr((goal_cable_5 - current_cable_5)/nb_step, type = "cable")
        incr_cable_6 = self._controlIncr((goal_cable_6 - current_cable_6)/nb_step, type = "cable")
        incr_cable_7 = self._controlIncr((goal_cable_7 - current_cable_7)/nb_step, type = "cable")
        incr_cable_8 = self._controlIncr((goal_cable_8 - current_cable_8)/nb_step, type = "cable")
        incr_base_pose = self._controlIncr((goal_base_pos - current_base_pos)/nb_step, type = "base")

        incr = [incr_cable_1, incr_cable_2, incr_cable_3, incr_cable_4,
                incr_cable_5, incr_cable_6, incr_cable_7, incr_cable_8,
                incr_base_pose]

        return incr

    def apply_action(self, incr):
        for i in range(8):
            self._pull(i, incr[i])
        self._moveBase(incr[-1])


    # def onAnimateBeginEvent(self, event):
    #     if self.step == 0:
    #         self.root.Reward.update()
    #         self.step+=1
    #     actions = [-1, -1, -1, -1, -1, -1, -1, -1, 1]
    #     nb_step = 30
    #     incr = self.compute_action(actions, nb_step)
    #     self.apply_action(incr)
    #
    #     print("Reward :", self.root.Reward.getReward())
    #     print("State :", len(getState(self.root)))

def action_to_command(actions, root, nb_step):
    incr = root.applyAction.compute_action(actions, nb_step)
    return incr


def startCmd(root, actions, duration):
    incr = action_to_command(actions, root, duration/root.dt.value + 1)
    startCmd_TrunkCube(root, incr, duration)


def startCmd_TrunkCube(rootNode, incr, duration):

    #Definition of the elements of the animation
    def executeAnimation(rootNode, incr, factor):
        rootNode.applyAction.apply_action(incr)

    #Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"rootNode": rootNode,
                    "incr": incr},
            duration=duration, mode="once"))


class Visualisation(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]

        self.current_time = 0
        self.list_time = []
        self.list_reward_real = []
        self.save_name = "../../../Proxy/Results/TrunkCube_Push_Real/"
        os.makedirs(self.save_name, exist_ok = True)

        self.actions = [[ 1., -1. , 1., -0.28707296, 1., 1., 1., -1., 1.],
                        [-0.6060756, -0.7226401, 1., 0.93343645, -1., 1., -0.6672112, -1., -1.],
                        [ 0.58742493, -1., 0.8199706, 0.717013, -1., 1., -1., -1, -1]]
        self.scale = 30

        self.current_idx = 0
        self.already_done = 0
        self.current_incr = None

    def onAnimateBeginEvent(self, event):
        if self.current_idx == 0:
            self.root.Reward.update()

        if self.already_done%self.scale == 0 and self.current_idx < len(self.actions):
            current_action = self.actions[self.current_idx]
            print(">> Took action n°", self.current_idx, " : ", current_action)
            self.current_incr = self.root.applyAction.compute_action(current_action, self.scale)
            self.current_idx+=1

        print(">>  STEP:", self.current_idx)
        _, reward = self.root.Reward.getReward()
        print("[INFO]  >> Reward in the real robot:", reward)

        self.root.applyAction.apply_action(self.current_incr)
        self.already_done+=1


        self.current_time+= 0.01
        if self.current_time < 0.01*self.scale*len(self.actions):
            self.list_time.append(self.current_time)
            self.list_reward_real.append(reward)

            with open(self.save_name+"time.txt", 'w') as f:
                json.dump(self.list_time, f)
            with open(self.save_name+"reward_real.txt", 'w') as f:
                json.dump(self.list_reward_real, f)
        else:
            exit()

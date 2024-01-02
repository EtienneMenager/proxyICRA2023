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
            # pos_trunk = self.root.sceneModerator.trunk.trunk.MechanicalObject.position.value[:, :3]
            # pos_colliCube = self.root.sceneModerator.cube.cube.Collision.MechanicalObject.position.value[:, :3]
            # face_pos = [pos_colliCube[[0, 1, 4, 5]].mean(axis=0), pos_colliCube[[1, 2, 5, 6]].mean(axis=0), pos_colliCube[[2, 3, 6, 7]].mean(axis=0), pos_colliCube[[3, 0, 7, 4]].mean(axis=0)]
            # dist_face = [self._compute_dist(self.goal_pos, p) for p in face_pos]
            # id = np.argmax(dist_face)
            #
            # opposite_face = np.array(face_pos[id])
            # current_dist_opposite_face_tips = float(np.min(np.linalg.norm(pos_trunk-opposite_face, axis = 1)))
            #
            # # pos_trunk = self.root.sceneModerator.trunk.trunk.MechanicalObject.position.value[:, :3]
            # # current_dist_cube_tips = float(np.min(np.linalg.norm(pos_trunk-pos_cube, axis = 1)))
            #
            # factor = [0.7, 0.3]
            # dist_factor = 50
            # reward = factor[0]*(self.init_dist - current_dist_cube_goal)/self.init_dist + factor[1]*min(1, dist_factor/current_dist_opposite_face_tips)

            # #For transfert
            reward = current_dist_cube_goal
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

        self.cube = kwargs["cube"]
        self.trunk = kwargs["trunk"]


    def getPos(self):
        pos_cube = self.cube.getPos()
        pos_trunk = self.trunk.getPos()
        return [pos_cube, pos_trunk]

    def setPos(self, pos):
        self.cube.setPos(pos[0])
        self.trunk.setPos(pos[1])


################################################################################


class History(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]

        self.use = False
        if kwargs["use"]:
            self.use = kwargs["use"]

        if kwargs["use_reward"]:
            self.use_reward = kwargs["use_reward"]
        else:
            self.use_reward = False


        self.waitingtime = kwargs["waitingtime"]
        self.idx = 0
        self.history_state = []
        self.history_reward = []

    def onAnimateEndEvent(self, event):
        if self.idx < self.waitingtime:
            self.idx+=1
        else:
            if self.use:
                state =  _getState(self.rootNode)
                self.history_state.append(state)

                if self.use_reward:
                    _, reward = getReward(self.rootNode)
                    self.history_reward.append(reward)

    def clear(self):
        self.history_state = []
        self.history_reward = []

def getInfos(root):
    infos = root.History.history_state
    reward = root.History.history_reward
    root.History.clear()
    return {"infos": infos, "reward": reward, "points": [[] for _ in range(len(reward))]}

def setInfos(root, infos):
    trunk = root.sceneModerator.trunk
    cube = root.sceneModerator.cube

    # barycentre = trunk.trunk.MechanicalObject.position.value[:, :3].mean(axis = 0)
    # translation = np.array(infos["barycentre"])-barycentre
    # with trunk.trunk.MechanicalObject.position.writeable() as pos:
    #     pos[:, 0]+= translation[0]

    cube.cube.MechanicalObject.position.value = infos["cube_pos"]


def getState(rootNode):
    # #For the bayesian optimisation

    # state = rootNode.History.history_state
    # rootNode.History.clear()
    # return state

    trunk = rootNode.sceneModerator.trunk
    cube = rootNode.sceneModerator.cube

    trunkPos = trunk.trunk.MechanicalObject.position.value[::2, :3].reshape(-1).tolist()
    cubPos = cube.cube.MechanicalObject.position.value[0, :3].tolist()
    goal = rootNode.GoalSetter.getGoal()

    state = trunkPos + cubPos + goal
    return state


def _getState(rootNode):
    # #For the bayesian optimisation
    # trunk = rootNode.sceneModerator.trunk
    # trunkPos = trunk.trunk.collision1.MechanicalObject.position.value[:, :3].tolist()
    # trunkPos += trunk.trunk.collision2.MechanicalObject.position.value[:, :3].tolist()
    # return trunkPos

    trunk = rootNode.sceneModerator.trunk
    return trunk.trunk.MechanicalObject.position.value[:, :3].tolist()


def getReward(rootNode, bounds = [5, 400], penality = 0): #NOTE: penality = 10 for the training
    reward, dist =  rootNode.Reward.getReward()

    # # # For transfert
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
        self.abstractTrunk = kwargs["trunk"]

        self.max_move = self.abstractTrunk.max_move
        self.max_flex_1 = self.abstractTrunk.max_flex_1
        self.max_flex_2 = self.abstractTrunk.max_flex_2
        self.max_angular_rate = self.abstractTrunk.max_angular_rate
        self.max_incr = self.abstractTrunk.max_incr

        self.a_move, self.b_move =  self.max_move/2 , self.max_move/2
        self.a_flex_1, self.b_flex_1 = self.max_flex_1, 0
        self.a_flex_2, self.b_flex_2 = self.max_flex_2, 0


        print(">>  Init done.")

    def onAnimateBeginEvent(self, event): # called at each begin of animation step
        translation = self.abstractTrunk.rest_shape.rest_shapeMO.position.value[0, 0]

        self.abstractTrunk.control.rigidBase.RigidBaseMO.position.value = [[translation, 0, 0, 0, 0, 0, 1]]
        self.abstractTrunk.control.rigidBase.RigidBaseMO.rest_position.value = [[translation, 0, 0, 0, 0, 0, 1]]

        p = self.abstractTrunk.control.rigidBase.MappedFrames.FramesMO.position.value[1:].tolist()
        self.abstractTrunk.trunk.MechanicalObject.rest_position.value = np.array(p)

    def _normalizedAction_to_action(self, action, type, n_beam = None):
        if type == "flex":
            if n_beam == 1:
                return self.a_flex_1*action + self.b_flex_1
            else:
                return self.a_flex_2*action + self.b_flex_2
        else:
            return self.a_move*action + self.b_move

    def _controlIncr(self, incr, type):
        max_incr = self.max_angular_rate if type == "flex" else self.max_incr

        if abs(incr)>max_incr:
            if incr>=0:
                incr = max_incr
            else:
                incr = -max_incr
        return incr

    def _moveBase(self, incr):
        controlMO = self.abstractTrunk.rest_shape.rest_shapeMO
        if controlMO.position.value[0,0] + incr >= self.max_move and controlMO.position.value[0,0] + incr <=0:
            with controlMO.position.writeable() as pos:
                pos[0,0]+=incr

        else:
            if controlMO.position.value[0,0] + incr < self.max_move:
                with controlMO.position.writeable() as pos:
                    pos[0,0]=self.max_move
            else:
                with controlMO.position.writeable() as pos:
                    pos[0,0]=0

    def _flex(self, numBeam, type, incr, n_beam = None):
        max_flex = self.max_flex_1 if n_beam == 1 else self.max_flex_2
        rateAngular = self.abstractTrunk.control.rateAngularDeform.rateAngularDeformMO.rest_position
        if abs(rateAngular.value[numBeam][type] + incr)<= max_flex:
            with rateAngular.writeable() as pos:
                pos[numBeam][type]+=incr
        else:
            if rateAngular.value[numBeam][type] + incr < -max_flex:
                with rateAngular.writeable() as pos:
                    pos[numBeam][type]= -max_flex
            else:
                with rateAngular.writeable() as pos:
                    pos[numBeam][type]= max_flex


    def compute_action(self, actions, nb_step):
        goal_beam1_flex1 = self._normalizedAction_to_action(actions[0], type = "flex", n_beam = 1)
        goal_beam1_flex2 = self._normalizedAction_to_action(actions[1], type = "flex", n_beam = 1)
        goal_beam2_flex1 = self._normalizedAction_to_action(actions[2], type = "flex", n_beam = 2)
        goal_beam2_flex2 = self._normalizedAction_to_action(actions[3], type = "flex", n_beam = 2)

        current_beam1_flex1 = self.abstractTrunk.control.rateAngularDeform.rateAngularDeformMO.rest_position.value.tolist()[0][1]
        current_beam1_flex2 = self.abstractTrunk.control.rateAngularDeform.rateAngularDeformMO.rest_position.value.tolist()[0][2]
        current_beam2_flex1 = self.abstractTrunk.control.rateAngularDeform.rateAngularDeformMO.rest_position.value.tolist()[3][1]
        current_beam2_flex2 = self.abstractTrunk.control.rateAngularDeform.rateAngularDeformMO.rest_position.value.tolist()[3][2]

        incr_beam1_flex1 = self._controlIncr((goal_beam1_flex1 - current_beam1_flex1)/nb_step, type = "flex")
        incr_beam1_flex2 = self._controlIncr((goal_beam1_flex2 - current_beam1_flex2)/nb_step, type = "flex")
        incr_beam2_flex1 = self._controlIncr((goal_beam2_flex1 - current_beam2_flex1)/nb_step, type = "flex")
        incr_beam2_flex2 = self._controlIncr((goal_beam2_flex2 - current_beam2_flex2)/nb_step, type = "flex")

        incr = [incr_beam1_flex1, incr_beam1_flex2, incr_beam2_flex1, incr_beam2_flex2]

        goal_base_pos = self._normalizedAction_to_action(actions[4], type = "move")
        current_base_pos = self.abstractTrunk.rest_shape.rest_shapeMO.position.value.tolist()[0][0]
        incr_base_pos = self._controlIncr((goal_base_pos - current_base_pos)/nb_step, type = "move")
        incr.append(incr_base_pos)

        return incr

    def apply_action(self, incr):

        for id in [0,1,2]:
            self._flex(id, 1, incr[0], n_beam = 1)
        for id in [0,1,2]:
            self._flex(id, 2, incr[1], n_beam = 1)
        for id in [3,4,5]:
            self._flex(id, 1, incr[2], n_beam = 2)
        for id in [3,4,5]:
            self._flex(id, 2, incr[3], n_beam = 2)
        self._moveBase(incr[4])

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

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

from common.utils import express_point

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

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """

        current_pos = self._computePos()
        reward = float(current_pos[0]-self.pred)
        # print("\n>> ABSOLUTE POS: ", current_pos[0] - self.init_pos[0])
        #self.pred = current_pos[0]

        return reward


    def update(self):
        """Update function.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        self.init_pos = [p for p in self._computePos()]
        self.pred = self._computePos()[0]

    def _computePos(self):
        abstractMultigait = self.rootNode.sceneModerator.abstractMultigait
        pos = abstractMultigait.multigait.MechanicalObject.position.value[1,:3]
        return pos


class goalSetter(Sofa.Core.Controller):
    """Compute the goal.

    Methods:
    -------
        __init__: Initialization of all arguments.
        update: Initialize the value of cost.

    Arguments:
    ---------
        goalMO: <MechanicalObject>
            The mechanical object of the goal.
        goalPos: coordinates
            The coordinates of the goal.

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

        self.goalMO=None
        if 'goalMO' in kwargs:
            self.goalMO = kwargs["goalMO"]
        self.goalPos = None
        if 'goalPos' in kwargs:
            self.goalPos = kwargs["goalPos"]

    def update(self):
        """Set the position of the goal.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        pass

    def set_mo_pos(self, goal):
        """Modify the goal.

        Not used here.
        """
        pass


class sceneModerator(Sofa.Core.Controller):
    """Compute the goal.

    Methods:
    -------
        __init__: Initialization of all arguments.
        getPos: get the position of elements in the scene.
        setPos: set the position of elements in the scene.

    Arguments:
    ---------
        objectMO: <MechanicalObject>
            The mechanical object of the objetcs.
        multigait: <Sofa Objects>
            multigait.

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

        self.abstractMultigait=None
        if kwargs["abstractMultigait"]:
            self.abstractMultigait = kwargs["abstractMultigait"]

    def getPos(self):
        """Retun the position of the mechanical object of interest.

        Parameters:
        ----------
            None

        Returns:
        -------
            _: list
                The position(s) of the object(s) of the scene.
        """
        return self.abstractMultigait.getPos()

    def setPos(self, pos):
        """Set the position of the mechanical object of interest.

        Parameters:
        ----------
            pos: list
                The position(s) of the object(s) of the scene.

        Returns:
        -------
            None.
        """
        self.abstractMultigait.setPos(pos)



################################################################################

def getState(rootNode):
    """Compute the state of the environment/agent.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        State: list of float
            The state of the environment/agent.
    """
    abstractMultigait = rootNode.sceneModerator.abstractMultigait
    p0 = [0, 0, 0] #abstractMultigait.multigait.MechanicalObject.position.value[1].tolist()[:3]
    p1 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[0].tolist()[:3])]
    p2 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[2].tolist()[:3])]
    p3 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[3].tolist()[:3])]
    p4 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[4].tolist()[:3])]
    p5 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[5].tolist()[:3])]
    p6 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[6].tolist()[:3])]
    p7 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[7].tolist()[:3])]
    p8 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[8].tolist()[:3])]
    p9 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[9].tolist()[:3])]
    p10 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[10].tolist()[:3])]
    p11 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[11].tolist()[:3])]
    p12 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[12].tolist()[:3])]
    p13 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[13].tolist()[:3])]
    p14 = [(p - p0[i]) for i, p in enumerate(abstractMultigait.multigait.MechanicalObject.position.value[14].tolist()[:3])]

    state = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11 + p12 + p13 + p14
    return state


def getReward(rootNode):
    """Compute the reward using Reward.getReward().

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        done, reward

    """
    r = rootNode.Reward.getReward()
    return False, r

def getPos(root):
    """Retun the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.

    Returns:
    -------
        position: list
            The position(s) of the object(s) of the scene.
    """
    position = root.sceneModerator.getPos()
    return position


def setPos(root, pos):
    """Set the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.
        pos: list
            The position(s) of the object(s) of the scene.

    Returns:
    -------
        None.

    Note:
    ----
        Don't forget to init the new value of the position.

    """
    root.sceneModerator.setPos(pos)

################################################################################

class applyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root =  kwargs["root"]

        if "abstractMultigait" in kwargs:
            print(">>  Init abstractMultigait...")
            self.abstractMultigait = kwargs["abstractMultigait"]
            self.multigait = self.abstractMultigait.multigait
        else:
            print(">>  No abstractMultigait ...")
            exit(1)

        if "beams" in kwargs:
            print(">>  Init beams...")
            self.beams = kwargs["beams"]
        else:
            print(">>  No beams ...")
            exit(1)

        if "factor" in kwargs:
            print(">>  Init factor...")
            self.factor = kwargs["factor"]
        else:
            print(">>  No factor ...")
            exit(1)

        self.max_flex_center = self.abstractMultigait.max_flex_center
        self.max_flex_leg = self.abstractMultigait.max_flex_leg
        self.type = 1

        self.a_flexion_leg, self.b_flexion_leg  = self.max_flex_leg/2, self.max_flex_leg/2
        self.a_flexion_center, self.b_flexion_center  = self.max_flex_center/2, self.max_flex_center/2
        print(">>  Init done.")

        self.count = 0


    def onAnimateBeginEvent(self, event): # called at each begin of animation step
        reverse_idx = [0, 9, 10, 11, 12, 13, 14]

        [beam_center_1, beam_center_2, beam_legFR, beam_legFL, beam_legBR, beam_legBL] = self.beams

        p = [beam_center_2.rigidBase.MappedFrames.FramesMO.position.value[-1].tolist()]
        p+= [beam_center_1.rigidBase.MappedFrames.FramesMO.position.value[1].tolist()]
        p+= [beam_center_1.rigidBase.MappedFrames.FramesMO.position.value[-1].tolist()]

        p+= beam_legFR.lastFrame_1.MappedFrames.FramesMO.position.value[2:].tolist()
        p+= beam_legFL.lastFrame_0.MappedFrames.FramesMO.position.value[2:].tolist()
        p+= beam_legBR.lastFrame_0.MappedFrames.FramesMO.position.value[2:].tolist()
        p+= beam_legBL.lastFrame_1.MappedFrames.FramesMO.position.value[2:].tolist()

        with self.multigait.MechanicalObject.rest_position.writeable() as pos:
            for i in range(len(p)):
                if not i in reverse_idx:
                    pos[i] = np.array(p[i])
                else:
                    _p = p[i][-4:]
                    q = [_p[-1], _p[-4], _p[-3], _p[-2]]
                    q = Quaternion(*q)
                    rot= list(q*Quaternion(0, 0, 0, 1))
                    new_p = list(p[i][:-4]) + rot[1:] + [rot[0]]
                    pos[i] = np.array(new_p)


    def _flex(self, beam, type, incr, center = False):
        section = 0
        if center:
            max_flex = self.max_flex_center
        else:
            max_flex = self.max_flex_leg

        if abs(beam.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][type]+incr)<=max_flex and beam.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][type]+incr>0:
            with beam.rateAngularDeform.rateAngularDeformMO.rest_position.writeable() as pos:
                pos[section][type]+= incr
            if abs(beam.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][type]) < 0.0001:
                return 2
            return 1
        else:
            return 0

    def _normalizedAction_to_action(self, nAction, center = False):
        if center:
            action = self.a_flexion_center*nAction + self.b_flexion_center - 0.0001
        else:
            action = self.a_flexion_leg*nAction + self.b_flexion_leg - 0.0001
        return action

    def compute_action(self, actions, nb_step):
        self.nb_step = nb_step
        section = 0
        flexion_center_goal = self._normalizedAction_to_action(actions[0], center = True)
        flexion_FR_goal = self._normalizedAction_to_action(actions[1])
        flexion_FL_goal = self._normalizedAction_to_action(actions[2])
        flexion_BR_goal = self._normalizedAction_to_action(actions[3])
        flexion_BL_goal = self._normalizedAction_to_action(actions[4])

        beam_center_1 = self.beams[0]
        beam_center_2 = self.beams[1]
        beam_FR = self.beams[2]
        beam_FL = self.beams[3]
        beam_BR = self.beams[4]
        beam_BL = self.beams[5]


        flexion_center_1 = beam_center_1.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][self.type]
        flexion_center_2 = beam_center_2.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][self.type]
        flexion_FR = beam_FR.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][self.type]
        flexion_FL = beam_FL.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][self.type]
        flexion_BR = beam_BR.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][self.type]
        flexion_BL = beam_BL.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][self.type]

        incr_flexion_center_1 = (flexion_center_goal - flexion_center_1)/nb_step
        incr_flexion_center_2 = (flexion_center_goal - flexion_center_2)/nb_step
        incr_flexion_FR= (flexion_FR_goal - flexion_FR)/nb_step
        incr_flexion_FL = (flexion_FL_goal - flexion_FL)/nb_step
        incr_flexion_BR = (flexion_BR_goal - flexion_BR)/nb_step
        incr_flexion_BL = (flexion_BL_goal - flexion_BL)/nb_step

        return [incr_flexion_center_1, incr_flexion_center_2, incr_flexion_FR, incr_flexion_FL, incr_flexion_BR, incr_flexion_BL]

    def apply_action(self, incr):
        incr_flexion_center_1, incr_flexion_center_2, incr_flexion_FR, incr_flexion_FL, incr_flexion_BR, incr_flexion_BL = incr

        beam_center_1 = self.beams[0]
        beam_center_2 = self.beams[1]
        beam_FR = self.beams[2]
        beam_FL = self.beams[3]
        beam_BR = self.beams[4]
        beam_BL = self.beams[5]

        if self.count < self.nb_step/self.factor:
            self._flex(beam_center_1, self.type, incr_flexion_center_1, center = True)
            self._flex(beam_center_2, self.type, incr_flexion_center_2, center = True)
            self._flex(beam_FR, self.type, self.factor*incr_flexion_FR)
            self._flex(beam_FL, self.type, self.factor*incr_flexion_FL)
            self._flex(beam_BR, self.type, self.factor*incr_flexion_BR)
            self._flex(beam_BL, self.type, self.factor*incr_flexion_BL)
        elif self.count > self.nb_step:
            self.count = 0
        self.count += 1

def action_to_command(actions, root, nb_step):
    """Link between Gym action (int) and SOFA command (displacement of cables).

    Parameters:
    ----------
        action: int
            The number of the action (Gym).
        root:
            The root of the scene.

    Returns:
    -------
        The command.
    """

    incr = root.applyAction.compute_action(actions, nb_step)
    return incr


def startCmd(root, actions, duration):
    """Initialize the command from root and action.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        action: int
            The action.
        duration: float
            Duration of the animation.

    Returns:
    ------
        None.

    """
    incr = action_to_command(actions, root, round(duration/root.dt.value+ 1))
    startCmd_MultiGaitRobot(root, incr, duration)


def startCmd_MultiGaitRobot(rootNode, incr, duration):
    """Initialize the command.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        incr:
            The elements of the commande.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
    """

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

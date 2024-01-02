# -*- coding: utf-8 -*-
"""Controller for the Abstraction of Jimmy.


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "March 8 2021"

import Sofa
import json
import numpy as np

class ControllerTrunkCube(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        self.trunk = kwargs["trunk"]
        self.task = kwargs["task"]
        self.incr = 2
        self.max_move = -200
        self.max_flex = 0.027
        self.angularRate = 0.001

        self.current_beam = 0
        self.current_idx = [0, 1, 2]

        self._incr = 0
        self.pos = []
        self.action = [0, 0, 0, 0, -1]

    def onAnimateBeginEvent(self, event): # called at each begin of animation step
        if self._incr == 0:
            self.root.Reward.update()
            self._incr+=1

        translation = self.trunk.rest_shape.rest_shapeMO.position.value[0, 0]
        self.trunk.control.rigidBase.RigidBaseMO.position.value = [[translation, 0, 0, 0, 0, 0, 1]]
        self.trunk.control.rigidBase.RigidBaseMO.rest_position.value = [[translation, 0, 0, 0, 0, 0, 1]]


        p = self.trunk.control.rigidBase.MappedFrames.FramesMO.position.value[1:].tolist()
        self.trunk.trunk.MechanicalObject.rest_position.value = np.array(p)

        reward = self.root.Reward.getReward()

        trunk = self.root.sceneModerator.trunk
        cube = self.root.sceneModerator.cube

        trunkPos = trunk.trunk.MechanicalObject.position.value[::5, :3].reshape(-1).tolist()
        cubPos = cube.cube.MechanicalObject.position.value[0, :3].tolist()
        goal = self.root.GoalSetter.getGoal()

        trunkPos = [p/200 if i%3 == 0 else p/200 for i, p in enumerate(trunkPos)]
        cubPos = [cubPos[0]/200, cubPos[1]/200, cubPos[2]/200]
        goal = [goal[0]/200, goal[1]/200] if len(goal)==2 else [goal[0]/200]

        state = trunkPos + cubPos + goal

        print("[INFO]  >> Reward:", reward)
        print("[INFO]  >> State:", state)
        print("[INFO]  >> Goal:", goal)
        print("[INFO]  >> Len state:", len(state))


    def _moveBase(self, incr):
        controlMO = self.trunk.rest_shape.rest_shapeMO
        if controlMO.position.value[0,0] + incr >= self.max_move and controlMO.position.value[0,0] + incr <=0:
            with controlMO.position.writeable() as pos:
                pos[0,0]+=incr

    def _flex(self, numBeam, type, incr):
        rateAngular = self.trunk.control.rateAngularDeform.rateAngularDeformMO.rest_position
        if abs(rateAngular.value[numBeam][type] + incr)<= self.max_flex:
            with rateAngular.writeable() as pos:
                pos[numBeam][type]+=incr


    def onKeypressedEvent(self, event):
        key = event['key']

        if self.task == 1:
            if key == "A":
                self._moveBase(-self.incr)
            if key == "B":
                self._moveBase(self.incr)

        if key == "L":
            self.current_beam = (self.current_beam+1)%2
            if self.current_beam == 0:
                self.current_idx = [0, 1, 2]
            else:
                self.current_idx = [3, 4, 5]


        if ord(key)== 18:
            for id in self.current_idx:
                self._flex(id, 2, self.angularRate)
        if ord(key)==20:
            for id in self.current_idx:
                self._flex(id, 2, -self.angularRate)
        if ord(key)== 21:
            for id in self.current_idx:
                self._flex(id, 1, self.angularRate)
        if ord(key)==19:
            for id in self.current_idx:
                self._flex(id, 1, -self.angularRate)



class ControllerTrunkCubeTotal(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]

        self.start_time = 30
        self.time_step = 60
        self.moveS = 45
        self.moveL = 30
        self.incr = 0

        self.trunk = self.root.Trunk.trunk
        self.pos = []

    def onAnimateBeginEvent(self, event): # called at each begin of animation step
        print(">>  Incr.")
        if self.incr < self.start_time:
            print(">>    No action:", self.incr, "/", self.start_time)
            self.incr+=1
        elif self.incr < self.start_time + self.time_step:
            # self.trunk.cableL3.cable.value = [self.trunk.cableL3.cable.value[0] + self.moveL/self.time_step]
            self.trunk.cableL1.cable.value = [self.trunk.cableL1.cable.value[0] + self.moveS/self.time_step]
            print(">>    Action:", self.incr, "/", self.start_time + self.time_step)
            self.incr+=1
        elif self.incr == self.start_time + self.time_step:
            # self.trunk.cableL0.cable.value = [0]
            # self.trunk.cableS0.cable.value = [0]
            self.incr+=1
            print(">>  STOP")

    def onAnimateEndEvent(self, event):
        if self.incr <= self.start_time + self.time_step:
            p = self.trunk.collision1.MechanicalObject.position.value.tolist()
            p += self.trunk.collision2.MechanicalObject.position.value.tolist()
            self.pos.append(p)
        if self.incr == self.start_time + self.time_step:
            with open("./real_points.txt", 'w') as outfile:
                json.dump(self.pos, outfile)
            print("[INFO] >>  Save the points.")



class Visualisation(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root =  kwargs["root"]
        self.actions = kwargs["actions"]
        self.scale = kwargs["scale"]

        self.current_idx = 0
        self.already_done = 0
        self.current_incr = None

    def onAnimateBeginEvent(self, event):
        if self.current_idx == 0:
            self.root.Reward.update()
        if self.already_done%self.scale == 0 and self.current_idx < len(self.actions):
            current_action = self.actions[self.current_idx]
            self.current_incr = self.root.applyAction.compute_action(current_action, self.scale)
            self.current_idx+=1

        print(">>  STEP:", self.current_idx)
        self.root.Reward.getReward()
        print(self.current_incr)
        self.root.applyAction.apply_action(self.current_incr)
        self.already_done+=1

# -*- coding: utf-8 -*-
"""Controller for the Abstraction of Multigait.


Units: mm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "March 8 2021"

import Sofa
import json
import numpy as np
from pyquaternion import Quaternion

from AbstractMultigaitToolbox import getState, getReward

class ControllerMultigait(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root =  kwargs["root"]

        if "multigait" in kwargs:
            print(">>  Init Multigait...")
            self.multigait = kwargs["multigait"]
        else:
            print(">>  No multigait ...")
            exit(1)

        if "beams" in kwargs:
            print(">>  Init beams...")
            self.beams = kwargs["beams"]
        else:
            print(">>  No multigait ...")
            exit(1)

        self.angularRate = 0.02
        self.beam = self.beams[0]
        self.max_flex_center =0.08
        self.max_flex_leg = 0.04
        self.max_flex = self.max_flex_center
        self.center = True
        self.x = 0

        self._forward = 0
        print(">>  Init done.")

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

            #print(max(getState(self.root)), getReward(self.root))

    def _flex(self, beam, type, incr):
        section = 0
        if abs(beam.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][type]+incr)<=self.max_flex:
            with beam.rateAngularDeform.rateAngularDeformMO.rest_position.writeable() as pos:
                pos[section][type]+= incr
            if abs(beam.rateAngularDeform.rateAngularDeformMO.rest_position.value[section][type]) < 0.0001:
                return 2
            return 1
        else:
            return 0

    def forward(self):
        type = 1
        if self._forward == 0:
            self.center = False
            self.max_flex = self.max_flex_leg
            ret = self._flex(self.beams[4], type, self.angularRate)
            self._flex(self.beams[5], type, self.angularRate)
            if ret == 0:
                self._forward = 1
        elif self._forward == 1:
            self.center = True
            self.max_flex = self.max_flex_center
            self._flex(self.beams[0], type, self.angularRate)
            ret = self._flex(self.beams[1], type, self.angularRate)
            if ret == 0:
                self._forward = 2
        elif self._forward == 2:
            self.center = False
            self.max_flex = self.max_flex_leg
            ret = self._flex(self.beams[2], type, self.angularRate)
            self._flex(self.beams[3], type, self.angularRate)
            if ret == 0:
                self._forward = 3
        elif self._forward == 3:
            self.center = False
            self.max_flex = self.max_flex_leg
            ret = self._flex(self.beams[4], type, -self.angularRate)
            self._flex(self.beams[5], type, -self.angularRate)
            if ret == 2:
                self._forward = 4
        elif self._forward == 4:
            self.center = True
            self.max_flex = self.max_flex_center
            self._flex(self.beams[0], type, -self.angularRate)
            ret = self._flex(self.beams[1], type, -self.angularRate)
            if ret == 2:
                self._forward = 5
        elif self._forward == 5:
            self.center = False
            self.max_flex = self.max_flex_leg
            ret = self._flex(self.beams[2], type, -self.angularRate)
            self._flex(self.beams[3], type, -self.angularRate)
            if ret == 2:
                self._forward = 0

    def onKeypressedEvent(self, event):
        key = event['key']

        if key == "0":
            print("Selected is 0: beam = beam_center")
            self.beam = self.beams[0]
            self.max_flex = self.max_flex_center
            self.center = True
        if key == "A":
            print("Selected is 1: beam = beam_FR")
            self.beam = self.beams[1]
            self.max_flex = self.max_flex_leg
            self.center = False
        if key == "B":
            print("Selected is 2: beam = beam_FL")
            self.beam = self.beams[2]
            self.max_flex =self.max_flex_leg
            self.center = False
        if key == "C":
            print("Selected is 3: beam = beam_BR")
            self.beam = self.beams[3]
            self.max_flex = self.max_flex_leg
            self.center = False
        if key == "D":
            print("Selected is 4: beam = beam_BL")
            self.beam = self.beams[4]
            self.max_flex = self.max_flex_leg
            self.center = False

        type = 1
        if ord(key) == 18:  #left
            self._flex(self.beam, type, self.angularRate)
        if ord(key) == 20:  #right
            self._flex(self.beam, type, -self.angularRate)

        if key == 'F':
            self.forward()

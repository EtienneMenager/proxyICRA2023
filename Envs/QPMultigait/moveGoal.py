# -*- coding: utf-8 -*-
"""Visualise the evolution of the scene in a runSofa way.

"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "May 4 2021"

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

import json
import Sofa
import numpy as np


class MoveGoal(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root =  kwargs["root"]
        self.positions = kwargs["positions"]

        self.current_idx = 0

    def onAnimateBeginEvent(self, event):
        if self.current_idx < len(self.positions):
            current_pos = self.positions[self.current_idx]
            self.root.Goals.GoalMO.position.value = np.array(current_pos)
            self.current_idx += 1


class MoveGoalMultigait(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root =  kwargs["root"]
        self.positions = kwargs["positions"]

        self.current_idx = 0

    def onAnimateBeginEvent(self, event, translation = [55.26272867193276, -53.71608077293722, 17.499841404638524]):
        if self.current_idx == 0:
            self.init_pos = self._compute_pos()[0]

        if self.current_idx < len(self.positions):
            # print("\nMOVE\n")
            current_pos = np.array(self.positions[self.current_idx])
            current_pos[:,0]+= translation[0]
            current_pos[:,1]+= translation[1]
            current_pos[:,2]+= translation[2]
            self.root.Goals.GoalMO.position.value = current_pos
            self.current_idx += 1

    def onAnimateEndEvent(self, event):
        error = self._compute_error()
        print(">> Error:", error)
        print(">> Distance:", self.init_pos - self._compute_pos()[0])

    def _compute_error(self):
        pos_goal = self.root.Goals.GoalMO.position.value
        pos_effectors = self.root.model.modelCollis.Effectors.EffectorMO.position.value
        return np.linalg.norm(pos_goal-pos_effectors, axis = 1).mean()

    def _compute_pos(self):
        pos_effectors = self.root.model.modelCollis.Effectors.EffectorMO.position.value
        pos_barycentre = pos_effectors.mean(axis = 0)
        return pos_barycentre

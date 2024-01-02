# -*- coding: utf-8 -*-
"""Create the scene with the Abstraction of Jimmy.


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

import os
import numpy as np

import sys
import importlib
import pathlib

from math import cos
from math import sin

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
meshPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
pathSceneFile = os.path.dirname(os.path.abspath(__file__))
from common.utils import createCosserat as cosserat
from common.utils import addRigidObject

class Trunk():
    def __init__(self, *args, **kwargs):
        self.max_move = -200
        self.base_size = [2, 25, 25]
        self.inverse = kwargs["inverse"]

        if self.inverse:
            try:
                self.init_forces = np.loadtxt(pathSceneFile + "/CablesForces.txt")
            except:
                print("[WARNING]  >> No init_pressure. (file not found: "+pathSceneFile + "/CablesForces.txt)")
                self.init_forces = [0 for _ in range(8)] + [[0, 0, 0]]

    def onEnd(self, rootNode, collisionGroup=1):
        print(">>  Init Trunk")
        totalTrunk = rootNode.addChild("Trunk")

        #ADD trunk
        self.trunk = totalTrunk.addChild('trunk')
        self.trunk.addObject('EulerImplicitSolver', name='odesolver', firstOrder=0, rayleighMass=0.1, rayleighStiffness=0.1)
        self.trunk.addObject('ShewchukPCGLinearSolver', name='linearSolver',iterations=500, tolerance=1.0e-18, preconditioners="precond")
        self.trunk.addObject('SparseLDLSolver', name='precond', template="CompressedRowSparseMatrixd")
        self.trunk.addObject("GenericConstraintCorrection", solverName = 'precond')
        self.trunk.addObject('MeshVTKLoader', name='loader', filename=meshPath+'trunk.vtk', rotation = [0, 90, 0])
        self.trunk.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.trunk.addObject('TetrahedronSetTopologyModifier')
        self.trunk.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')
        self.trunk.addObject('MechanicalObject', name='tetras', rest_position="@loader.position", position="@loader.position", template='Vec3d', showIndices='false', showIndicesScale=4e-5)
        self.trunk.addObject('UniformMass', totalMass=0.042)
        self.trunk.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', poissonRatio=0.3,  youngModulus=3000)

        self.trunk.addObject('BoxROI', name='boxROI', box=[-2, -25, -25, 4, 25, 25], drawBoxes=True)
        self.trunk.boxROI.init()

        self.trunk.addObject('PartialFixedConstraint', fixedDirections=[0, 1, 1, 1, 1, 1], indices="@boxROI.indices")
        actuator = self.trunk.addChild('actuator_base')
        self.sliding = actuator.addObject('MechanicalObject', template = 'Vec3d', position =  self.trunk.boxROI.pointsInROI.value)
        actuator.addObject('BarycentricMapping')
        actuator.addObject('RestShapeSpringsForceField', name='control', stiffness=1e12)

        #ADD Base
        base = self.trunk.addChild("Base")
        topo = base.addObject('MeshOBJLoader', name = "cube_topo", filename=meshPath+'cube.obj', scale3d=self.base_size)
        base.addObject('TriangleSetTopologyContainer', src=topo.getLinkPath())
        base.addObject('MechanicalObject')
        base.addObject('TriangleCollisionModel', group=collisionGroup)
        base.addObject('LineCollisionModel', group=collisionGroup)
        base.addObject('PointCollisionModel', group=collisionGroup)
        base.addObject('PartialFixedConstraint', fixedDirections=[0, 1, 1, 1, 1, 1])
        base.addObject('BarycentricMapping')

    	# trunk/cables
        self._add_cable(self.trunk)

    	#Add Collision
        for i in range(1,3):
            trunkCollision = self.trunk.addChild('collision'+str(i))
            trunkCollision.addObject('MeshSTLLoader', name="loader", filename=meshPath+"trunk_colli"+str(i)+".stl", rotation = [0, 90, 0])
            trunkCollision.addObject('MeshTopology', src="@loader")
            trunkCollision.addObject('MechanicalObject')
            trunkCollision.addObject('TriangleCollisionModel', group=collisionGroup)
            trunkCollision.addObject('LineCollisionModel', group=collisionGroup)
            trunkCollision.addObject('PointCollisionModel', group=collisionGroup)
            trunkCollision.addObject('BarycentricMapping')

        #Add visu
        trunkVisu = self.trunk.addChild('visu')
        trunkVisu.addObject('MeshSTLLoader', name="loader", filename=meshPath+"trunk.stl")
        trunkVisu.addObject('OglModel', template='Vec3d', src="@loader", color=[1., 1., 1., 1.], rotation = [0, 90, 0])
        trunkVisu.addObject('BarycentricMapping')



    def _rotate(self, v,q):
        c0 = ((1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))*v[0] + (2.0 * (q[0] * q[1] - q[2] * q[3])) * v[1] + (2.0 * (q[2] * q[0] + q[1] * q[3])) * v[2])
        c1 = ((2.0 * (q[0] * q[1] + q[2] * q[3]))*v[0] + (1.0 - 2.0 * (q[2] * q[2] + q[0] * q[0]))*v[1] + (2.0 * (q[1] * q[2] - q[0] * q[3]))*v[2])
        c2 = ((2.0 * (q[2] * q[0] - q[1] * q[3]))*v[0] + (2.0 * (q[1] * q[2] + q[0] * q[3]))*v[1] + (1.0 - 2.0 * (q[1] * q[1] + q[0] * q[0]))*v[2])

        return [c0, c1, c2]

    def _normalize(self, x):
        norm = np.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
        for i in range(0,3):
            x[i] = x[i]/norm

    def _add_cable(self, trunk):
        max_force = 60

        length1 = 10
        length2 = 2
        lengthTrunk = 195

        pullPoint = [[0., length1, 0.], [-length1, 0., 0.], [0., -length1, 0.], [length1, 0., 0.]]
        direction = [0, length2-length1, lengthTrunk]
        self._normalize(direction)

        displacementL = [7.62, -18.1, 3.76, 30.29]
        displacementS = [-0.22, -7.97, 3.89, 12.03]

        nbCables = 4
        self.cables = []

        for i in range(0,nbCables):
            theta = 1.57*i
            q = [0.,0.,sin(theta/2.), cos(theta/2.)]

            position = [[0, 0, 0]]*20
            for k in range(0,20,2):
                v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21]
                position[k] = self._rotate(v,q)
                v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27]
                position[k+1] = self._rotate(v,q)

            pullPointList = [[pullPoint[i][0], pullPoint[i][1], pullPoint[i][2]]]

            cableL = trunk.addChild('cableL'+str(i))
            cableL.addObject('MechanicalObject', name='meca',position= pullPointList+ position, rotation = [0, 90, 0])

            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            if self.inverse:
                cable = cableL.addObject('CableActuator', template='Vec3', name='cable', indices=idx, hasPullPoint=0, minForce=0, initForce = self.init_forces[i], maxForce = max_force)
            else:
                cable = cableL.addObject('CableConstraint', template='Vec3d', name="cable", hasPullPoint= 0, indices= idx, valueType="displacement", minForce = 0)
            self.cables.append(cable)
            cableL.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

            # pipes
            pipes = trunk.addChild('pipes'+str(i))
            pipes.addObject('EdgeSetTopologyContainer', position= pullPointList + position, edges= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            pipes.addObject('MechanicalObject', name="pipesMO", rotation = [0, 90, 0])
            pipes.addObject('UniformMass', totalMass=0.003)
            pipes.addObject('MeshSpringForceField', stiffness=1.5e2, damping=0, name="FF")
            pipes.addObject('BarycentricMapping', name="BM")


        for i in range(0,nbCables):
            theta = 1.57*i
            q = [0.,0.,sin(theta/2.), cos(theta/2.)]

            position = [[0, 0, 0]]*10
            for k in range(0,9,2):
                v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21]
                position[k] = self._rotate(v,q)
                v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27]
                position[k+1] = self._rotate(v,q)

            pullPointList = [[pullPoint[i][0], pullPoint[i][1], pullPoint[i][2]]]

            cableS = trunk.addChild('cableS'+str(i))
            cableS.addObject('MechanicalObject', name='meca', position=pullPointList+ position, rotation = [0, 90, 0])

            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            if self.inverse:
                cable = cableS.addObject('CableActuator', template='Vec3', name='cable', indices=idx, hasPullPoint=0, minForce = 0, initForce = self.init_forces[i+nbCables], maxForce = max_force)
            else:
                cable = cableS.addObject('CableConstraint', template='Vec3d', name="cable", hasPullPoint=0, indices=idx, valueType="displacement", value=0,  minForce = 0)
            self.cables.append(cable)
            cableS.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    def getPos(self):
        return self.trunk.tetras.position.value.tolist()

    def setPos(self, pos):
        self.trunk.tetras.position.value = np.array(pos)


class Cube():
    def __init__(self, *args, **kwargs):

        if "cube_config" in kwargs:
            print(">>  Init cube_config...")
            self.cube_config = kwargs["cube_config"]
            self.init_pos = self.cube_config["init_pos"]
            self.density = self.cube_config["density"]
            self.scale = self.cube_config["scale"]
        else:
            print(">>  No cube_config ...")
            exit(1)

    def onEnd(self, rootNode, collisionGroup = 1):
        print(">>  Init Cube")
        addRigidObject(rootNode,filename=meshPath+'cube.obj',name='Cube',scale=self.scale, position=self.init_pos+[0, 0, 0, 1], density=self.density, collisionGroup = collisionGroup)
        self.cube = rootNode.Cube
        # self.cube.addObject('PartialFixedConstraint', fixedDirections=[0, 0, 0, 1, 0, 1], indices="@boxROI.indices")


    def getPos(self):
        posCube = self.cube.MechanicalObject.position.value.tolist()
        return [posCube]

    def setPos(self, pos):
        [posCube] = pos
        self.cube.MechanicalObject.position.value = np.array(posCube)

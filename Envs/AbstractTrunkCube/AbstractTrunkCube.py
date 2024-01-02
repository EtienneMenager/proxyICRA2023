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
        self.trunk_config = kwargs["trunk_config"]
        self.max_move = self.trunk_config["max_move"]
        self.base_size = self.trunk_config["base_size"]
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

        self.trunk.addObject('BoxROI', name='boxROI', box=[-10, -10, 0, 10, 10, 10], drawBoxes=True)
        self.trunk.boxROI.init()

        self.trunk.addObject('PartialFixedConstraint', fixedDirections=[0, 1, 1, 1, 1, 1], indices="@boxROI.indices")
        actuator = self.trunk.addChild('actuator_base')
        self.sliding = actuator.addObject('MechanicalObject', template = 'Vec3d', position =  self.trunk.boxROI.pointsInROI.value)
        if self.inverse:
            actuator.addObject('SlidingActuator', template='Vec3d', name="actuator_base", direction = [1, 0, 0, 0, 0, 0],
                                         maxDispVariation=5,
                                        initDisplacement  = 0, indices = [i for i in range(len(self.trunk.boxROI.pointsInROI.value.tolist()))])
        actuator.addObject('BarycentricMapping')


        #ADD Base
        base = self.trunk.addChild("Base")
        topo = base.addObject('MeshOBJLoader', name = "cube_topo", filename=meshPath+'cube.obj', scale3d=self.base_size)
        base.addObject('TriangleSetTopologyContainer', src=topo.getLinkPath())
        base.addObject('MechanicalObject')
        base.addObject('TriangleCollisionModel', group=collisionGroup)
        base.addObject('LineCollisionModel', group=collisionGroup)
        base.addObject('PointCollisionModel', group=collisionGroup)
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


class AbstractTrunk():
    def __init__(self, *args, **kwargs):

        if "cosserat_config" in kwargs:
            print(">>  Init cosserat_config...")
            self.cosserat_config = kwargs["cosserat_config"]
        else:
            print(">>  No cosserat_config ...")
            exit(1)


        if "trunk_config" in kwargs:
            print(">>  Init trunk_config...")
            self.trunk_config = kwargs["trunk_config"]

            self.max_flex_1 = self.trunk_config["max_flex"][0]
            self.max_flex_2 = self.trunk_config["max_flex"][1]
            self.max_move = self.trunk_config["max_move"]
            self.base_size = self.trunk_config["base_size"]
            self.max_angular_rate = self.trunk_config["max_angular_rate"]
            self.max_incr = self.trunk_config["max_incr"]
            self.young_modulus = self.trunk_config["young_modulus"]
            self.mass = self.trunk_config["mass"]

        else:
            print(">>  No trunk_config ...")
            exit(1)

    def onEnd(self, rootNode, collisionGroup = 1):
        print(">>  Init Trunk")
        totalTrunk = rootNode.addChild("Trunk")

        #ADD cosserat
        self.control = cosserat(totalTrunk, self.cosserat_config, name = "control", orientation = [0, 0, 0, 1], radius = 0.5)

        #ADD trunk
        self.trunk = totalTrunk.addChild("Trunk")

        positions = [[i*self.cosserat_config['tot_length']/self.cosserat_config['nbFramesF'], 0, 0, 0, 0, 0, 1] for i in range(self.cosserat_config['nbFramesF']+1)]
        edges = [[i, i+1] for i in range(self.cosserat_config['nbFramesF'])]
        self.trunk.addObject('EulerImplicitSolver')
        self.trunk.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixd", name='solver')
        self.trunk.addObject('EdgeSetTopologyContainer', position=positions, edges=edges)
        self.trunk.addObject('MechanicalObject', template='Rigid3', position=[x for x in positions])

        mass = self.trunk.addChild("Mass")
        a, b = (self.mass[1]-self.mass[0])/self.cosserat_config['nbFramesF'], self.mass[0]
        value_mass = [a*i+b for i in range(self.cosserat_config['nbFramesF']+1)]
        for i in range(self.cosserat_config['nbFramesF']+1):
            mass.addObject('UniformMass', name = "UniformMass_"+str(i), totalMass=value_mass[i], indices = [i], context="@../")

        self.trunk.addObject('PartialFixedConstraint', fixedDirections=[0, 1, 1, 1, 1, 1], indices = 0)

        radius_1, radius_2 =  10, 10
        interpolation_1 = self.trunk.addObject('BeamInterpolation', name = 'BeamInterpolation_1', defaultYoungModulus=self.young_modulus[0], straight=False, crossSectionShape = 'circular', radius = radius_1, edgeList = edges[:len(edges)//2])
        interpolation_2 = self.trunk.addObject('BeamInterpolation', name = 'BeamInterpolation_2',  defaultYoungModulus=self.young_modulus[1], straight=False, crossSectionShape = 'circular', radius = radius_2, edgeList = edges[len(edges)//2:])
        self.trunk.addObject('AdaptiveBeamForceFieldAndMass', name='AdaptiveBeamForceFieldAndMass_1', massDensity= 0, computeMass = False, interpolation = interpolation_1.getLinkPath())
        self.trunk.addObject('AdaptiveBeamForceFieldAndMass', name='AdaptiveBeamForceFieldAndMass_2', massDensity= 0, computeMass = False, interpolation = interpolation_2.getLinkPath())
        self.trunk.addObject('LinearSolverConstraintCorrection')

        #ADD Base
        base = self.trunk.addChild("Base")
        topo = base.addObject('MeshOBJLoader', name = "cube_topo", filename=meshPath+'cube.obj', scale3d=self.base_size)
        base.addObject('TriangleSetTopologyContainer', src=topo.getLinkPath())
        base.addObject('MechanicalObject')
        base.addObject('TriangleCollisionModel', group=collisionGroup)
        base.addObject('LineCollisionModel', group=collisionGroup)
        base.addObject('PointCollisionModel', group=collisionGroup)
        base.addObject('RigidMapping', index=0)

        #Visu:
        trunkVisu = self.trunk.addChild("visu")
        trunkVisu.addObject('MeshSTLLoader', name="loader", filename=meshPath+"trunk.stl")
        trunkVisu.addObject('OglModel', template='Vec3d', src="@loader", color=[1., 1., 1., 1.], rotation = [0, 90, 0])
        trunkVisu.addObject('SkinningMapping')

        #Colli:
        for i in range(1,3):
            trunkCollision = self.trunk.addChild('collision'+str(i))
            trunkCollision.addObject('MeshSTLLoader', name="loader", filename=meshPath+"trunk_colli"+str(i)+".stl", rotation = [0, 90, 0])
            trunkCollision.addObject('MeshTopology', src="@loader")
            trunkCollision.addObject('MechanicalObject')
            trunkCollision.addObject('TriangleCollisionModel', group=collisionGroup)
            trunkCollision.addObject('LineCollisionModel', group=collisionGroup)
            trunkCollision.addObject('PointCollisionModel', group=collisionGroup)
            trunkCollision.addObject('SkinningMapping')

        self.rest_shape = totalTrunk.addChild("rest_shape")
        rest_shapeMO = self.rest_shape.addObject("MechanicalObject", name = "rest_shapeMO", template = 'Rigid3', position= [0, 0, 0, 0, 0, 0, 1])
        self.trunk.addObject('RestShapeSpringsForceField', name='control', points=0,
                           external_rest_shape=rest_shapeMO.getLinkPath(), stiffness=1e12)

    def getPos(self):
        poseTrunk = self.trunk.MechanicalObject.position.value[:].tolist()
        poseControl = self.rest_shape.rest_shapeMO.position.value[:].tolist()

        posCosserat = self.control.rigidBase.MappedFrames.FramesMO.position.value[:].tolist()
        posOutputCosserat = self.control.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rateCosserat = self.control.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()

        collision_1 = self.trunk.collision1.MechanicalObject.position.value[:].tolist()
        collision_2 = self.trunk.collision2.MechanicalObject.position.value[:].tolist()

        return [poseTrunk, poseControl, posCosserat, posOutputCosserat, rateCosserat, collision_1, collision_2]

    def setPos(self, pos):
        poseTrunk, poseControl, posCosserat, posOutputCosserat, rateCosserat, _, _ = pos

        self.trunk.MechanicalObject.position.value = np.array(poseTrunk)
        self.rest_shape.rest_shapeMO.position.value = np.array(poseControl)

        self.control.rigidBase.MappedFrames.FramesMO.position.value = np.array(posCosserat)
        self.control.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutputCosserat)
        self.control.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rateCosserat)
        self.control.rigidBase.MappedFrames.DiscreteCosseratMapping.init()


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

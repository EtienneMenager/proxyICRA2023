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
from pyquaternion import Quaternion

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")

from common.utils import createCosserat as cosserat

class OptiAbstractMultigait():
    def __init__(self, *args, **kwargs):

        if "cosserat_config" in kwargs:
            print(">>  Init cosserat_config...")
            self.cosserat_config = kwargs["cosserat_config"]
        else:
            print(">>  No cosserat_config ...")
            exit(1)

        if "multigait_config" in kwargs:
            print(">>  Init multigait_config...")
            self.multigait_config = kwargs["multigait_config"]
        else:
            print(">>  No multigait_config ...")
            exit(1)

        init_pos= self.multigait_config["init_pos"]
        self.mass_density = self.multigait_config["mass_density"]
        self.young_modulus = self.multigait_config["young_modulus"]
        self.max_flex_center = self.multigait_config["max_flex_center"]
        self.max_flex_leg = self.multigait_config["max_flex_leg"]

        nbSectionS_leg= self.cosserat_config["nbSectionS_leg"]
        nbFramesF_leg= self.cosserat_config["nbFramesF_leg"]
        nbSectionS_center= self.cosserat_config["nbSectionS_center"]
        nbFramesF_center= self.cosserat_config["nbFramesF_center"]

        self.config_leg = {'init_pos': init_pos, 'nbSectionS': nbSectionS_leg,
                         'tot_length': 69, 'nbFramesF': nbFramesF_leg, }

        self.config_center = {'init_pos': init_pos, 'nbSectionS': nbSectionS_center,
                            'tot_length': 59, 'nbFramesF': nbFramesF_center}

        self.n_cube_center = self.config_center['nbFramesF']
        self.n_cube_leg = self.config_leg['nbFramesF']

    def onEnd(self, rootNode, radius = 0.5):
        print(">>  Init Multigait...")
        self.multigait= rootNode.addChild('Multigait')


        self.config_center['init_pos'][0] = self.config_center['init_pos'][0]+ self.config_center['tot_length']/2
        self.config_center['tot_length'] = self.config_center['tot_length']/2

        ang_g, ang_d = 20, -20
        quat_g, quat_d = Quaternion(axis=[0, 0, 1], degrees=ang_g), Quaternion(axis=[0, 0, 1], degrees=ang_d)
        v_quat_g, v_quat_d, s_quat_g, s_quat_d = quat_g.vector, quat_d.vector, quat_g.scalar, quat_d.scalar
        orient = [[v_quat_g[0], v_quat_g[1], v_quat_g[2], s_quat_g], [v_quat_d[0], v_quat_d[1], v_quat_d[2], s_quat_d]]

        lastFrame = {"orient": orient, "index": 2, "dist": 1}
        self.beam_center_1 = cosserat(rootNode, self.config_center, name = "beam_center_1", orientation = [0,0, 0, 1], radius = radius, last_frame=lastFrame, youngModulus=self.young_modulus[1])
        self.beam_center_2 = cosserat(rootNode, self.config_center, name = "beam_center_2", orientation = [0,0, 1, 0], radius = radius, last_frame=lastFrame, youngModulus=self.young_modulus[1])


        pos_x, pos_y = self.config_center['tot_length'], self.config_center['tot_length']*np.tan((ang_g*2*np.pi)/360)
        self.config_leg['init_pos'][0], self.config_leg['init_pos'][1] = 2*pos_x, pos_y
        self.config_leg['tot_length'] = self.config_leg['tot_length'] - np.sqrt(pos_x**2 + pos_y**2)
        self.beam_legFL = cosserat(rootNode, self.config_leg, name = "beam_legFL", orientation = orient[0], radius = radius, youngModulus=self.young_modulus[0])

        self.config_leg['init_pos'][0], self.config_leg['init_pos'][1] = 2*pos_x, -pos_y
        self.beam_legFR = cosserat(rootNode, self.config_leg, name = "beam_legFR", orientation = orient[1], radius = radius, youngModulus=self.young_modulus[0])

        self.config_leg['init_pos'][0], self.config_leg['init_pos'][1] = 2*pos_x, -pos_y
        self.beam_legBL = cosserat(rootNode, self.config_leg, name = "beam_legBL", radius = radius, orientation =orient[1], youngModulus=self.young_modulus[0])
        self.config_leg['init_pos'][0], self.config_leg['init_pos'][1] = 2*pos_x, pos_y
        self.beam_legBR = cosserat(rootNode, self.config_leg, name = "beam_legBR", radius = radius,  orientation =orient[0], youngModulus=self.young_modulus[0])


        MappingCommand = rootNode.addChild("MappingCommand")
        mapping_legFL = MappingCommand.addChild("mapping_legFL")
        mapping_legFL.addObject("RigidRigidMapping", name = "mapping_legFL",
                            input = self.beam_center_1.rigidBase.MappedFrames.FramesMO.getLinkPath(),
                            output = self.beam_legFL.rigidBase.RigidBaseMO.getLinkPath(), index = 4,
                            globalToLocalCoords = True)

        mapping_legFR = MappingCommand.addChild("mapping_legFR")
        mapping_legFR.addObject("RigidRigidMapping", name = "mapping_legFR",
                            input = self.beam_center_1.rigidBase.MappedFrames.FramesMO.getLinkPath(),
                            output = self.beam_legFR.rigidBase.RigidBaseMO.getLinkPath(), index = 4,
                            globalToLocalCoords = True)

        mapping_legBL = MappingCommand.addChild("mapping_legBL")
        mapping_legBL.addObject("RigidRigidMapping", name = "mapping_legBL",
                            input = self.beam_center_2.rigidBase.MappedFrames.FramesMO.getLinkPath(),
                            output = self.beam_legBL.rigidBase.RigidBaseMO.getLinkPath(), index = 4,
                            globalToLocalCoords = True)

        mapping_legBR = MappingCommand.addChild("mapping_legBR")
        mapping_legBR.addObject("RigidRigidMapping", name = "mapping_legBR",
                            input = self.beam_center_2.rigidBase.MappedFrames.FramesMO.getLinkPath(),
                            output = self.beam_legBR.rigidBase.RigidBaseMO.getLinkPath(), index = 4,
                            globalToLocalCoords = True)


        pos_beam_center_1, idx_beam_center_1, same_direction_beam_center_1 = [[29.5, 0, 0, 0, 0, 0, 1], [49.1667, 0, 0, 0, 0, 0, 1], [59, 0, 0, 0, 0, 0, 1]], [[0, 3, 4]], [[True, True, True]]
        pos_beam_center_2, idx_beam_center_2, same_direction_beam_center_2 = [[9.83333, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]], [[3, 4]], [[False, False]]
        pos_beam_legFL, idx_beam_legFL, same_direction_beam_legFL = [[59, 10.7371, 0, 0, 0, 0.173648, 0.984808], [70.7796, 15.0245, 0, 0, 0, 0.173648, 0.984808], [82.5592, 19.312, 0, 0, 0, 0.173648, 0.984808], [94.3388, 23.5994, 0, 0, 0, 0.173648, 0.984808]], [[0, 2, 3, 4]], [[True, True, True, True]]
        pos_beam_legFR, idx_beam_legFR, same_direction_beam_legFR = [[59, -10.7371, 0, 0, 0, -0.173648, 0.984808], [70.7796, -15.0245, 0, 0, 0, -0.173648, 0.984808], [82.5592, -19.312, 0, 0, 0, -0.173648, 0.984808], [94.3388, -23.5994, 0, 0, 0, -0.173648, 0.984808]], [[0, 2, 3, 4]], [[True, True, True, True]]
        pos_beam_legBL, idx_beam_legBL, same_direction_beam_legBL = [[-4.44089e-16, 10.7371, 0, 0, 0, 0.173648, -0.984808], [-11.7796, 15.0245, 0, 0, 0, 0.173648, -0.984808], [-23.5592, 19.312, 0, 0, 0, 0.173648, -0.984808], [-35.3388, 23.5994, 0, 0, 0, 0.173648, -0.984808]], [[0, 2, 3, 4]], [[False, False, False, False]]
        pos_beam_legBR, idx_beam_legBR, same_direction_beam_legBR = [[-4.44089e-16, -10.7371, 0, 0, 0, -0.173648, -0.984808], [-11.7796, -15.0245, 0, 0, 0, -0.173648, -0.984808], [-23.5592, -19.312, 0, 0, 0, -0.173648, -0.984808], [-35.3388, -23.5994, 0, 0, 0, -0.173648, -0.984808]], [[0, 2, 3, 4]], [[False, False, False, False]]

        pos = pos_beam_center_1 + pos_beam_center_2 + pos_beam_legFL + pos_beam_legFR + pos_beam_legBL + pos_beam_legBR
        idx = idx_beam_center_1 + idx_beam_center_2 + idx_beam_legFL + idx_beam_legFR + idx_beam_legBL + idx_beam_legBR
        same_direction = same_direction_beam_center_1 + same_direction_beam_center_2 + same_direction_beam_legFL + same_direction_beam_legFR + same_direction_beam_legBL + same_direction_beam_legBR
        beams = [self.beam_center_1, self.beam_center_2, self.beam_legFL, self.beam_legFR, self.beam_legBL, self.beam_legBR]

        edges = [[0,1], [1,2], [4,3], [3,0], #center
                 [0,5], [5,6], [6,7], [7,8], #FL
                 [0,9], [9,10], [10,11], [11,12], #FR
                 [16,15], [15,14], [14,13], [13,0], #BL
                 [20,19], [19,18], [18,17], [17,0],  #BR
                 [2, 5], [2, 9], [13, 4], [17, 4]]


        self._addMultigait(rootNode, pos, edges, collisionGroup=0)

        return idx, same_direction, beams


    def _addMultigait(self, node, positions, edges, collisionGroup=0):
        barycentre = [ -76.72398876-self.config_center['tot_length'] , 53.71639887,  0] # -3.17560985]
        barycentre[0] = barycentre[0] + self.config_center['init_pos'][0]
        lengthY_center, lengthY_leg, lengthZ = 30, 15, 6

        self.multigait.addObject('EulerImplicitSolver')
        self.multigait.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixd", name='solver')
        self.multigait.addObject('EdgeSetTopologyContainer', position=positions, edges=edges)
        self.multigait.addObject('MechanicalObject', template='Rigid3', position=[x for x in positions])

        edgeList_legs = [5, 6, 7,9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23]
        lengthList_legs = [12.5356]*(len(edgeList_legs)-4) + [12.4375, 12.4375, 12.4375, 12.4375]
        DOF0TransformNode0_legs = [[0, 0, 0, 0, 0, 0, 1]]*len(edgeList_legs)
        DOF1TransformNode1_legs = [[0, 0, 0, 0, 0, 0, 1]]*len(edgeList_legs)
        interpolation_legs = self.multigait.addObject('BeamInterpolation', name = 'BeamInterpolation_legs',lengthY=lengthY_leg, lengthZ=lengthZ, defaultYoungModulus=self.young_modulus[0], straight=False, crossSectionShape = 'rectangular',
                                    edgeList = edgeList_legs, lengthList = lengthList_legs, DOF0TransformNode0 = DOF0TransformNode0_legs, DOF1TransformNode1 = DOF1TransformNode1_legs)
        self.multigait.addObject('AdaptiveBeamForceFieldAndMass', name='AdaptiveBeamForceFieldAndMass_legs', massDensity= self.mass_density[0], interpolation = interpolation_legs.getLinkPath())

        edgeList_center = [0, 1, 2, 3, 4, 8, 15, 19]
        lengthList_center =  [19.6667, 9.8333, 9.8333, 19.6667, 31.6448, 31.6448, 31.6448, 31.6448]
        DOF0TransformNode0_center = [[0, 0, 0, 0, 0, 0, 1]]*len(edgeList_center)
        DOF1TransformNode1_center = [[0, 0, 0, 0, 0, 0, 1]]*len(edgeList_center)
        interpolation_center = self.multigait.addObject('BeamInterpolation', name = 'BeamInterpolation_center', lengthY=lengthY_center, lengthZ=lengthZ, defaultYoungModulus=self.young_modulus[1], straight=False, crossSectionShape = 'rectangular',
                                    edgeList = edgeList_center, lengthList = lengthList_center, DOF0TransformNode0 = DOF0TransformNode0_center, DOF1TransformNode1 = DOF1TransformNode1_center)
        self.multigait.addObject('AdaptiveBeamForceFieldAndMass', name='AdaptiveBeamForceFieldAndMass_center', massDensity= self.mass_density[1], interpolation = interpolation_center.getLinkPath())


        self.multigait.addObject('LinearSolverConstraintCorrection')

        pathMesh = os.path.dirname(os.path.abspath(__file__))+'/mesh'
        modelCollis = self.multigait.addChild('modelCollis')
        modelCollis.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'/quadriped_collision.stl',
                              rotation=[0, 0, 0], translation=barycentre)
        modelCollis.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
        modelCollis.addObject('MechanicalObject', name='collisMO', template='Vec3d')
        modelCollis.addObject('TriangleCollisionModel', group=collisionGroup)
        modelCollis.addObject('LineCollisionModel', group=collisionGroup)
        modelCollis.addObject('PointCollisionModel', group=collisionGroup)
        modelCollis.addObject('SkinningMapping')

        modelCollis.addObject('BoxROI', name='membraneROISubTopo', box=[0+barycentre[0], 0+barycentre[1], -0.1+barycentre[2], 150+barycentre[0], -100+barycentre[1], 0.1+barycentre[2]], computeTetrahedra=False,
                        drawBoxes=True)
        self.effectors = modelCollis.addChild("Effectors")
        self.effectors.addObject('PointSetTopologyContainer', position='@membraneROISubTopo.pointsInROI', name ="container")
        self.effectors.addObject('MechanicalObject', showObject = True, showObjectScale=10, showColor=[0, 0, 1, 1])
        self.effectors.addObject('BarycentricMapping')

        return self.multigait

    def getPos(self):
        posBase_center_1 = self.beam_center_1.rigidBase.RigidBaseMO.position.value[:].tolist()
        posFrame_center_1 = self.beam_center_1.rigidBase.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_center_1 = self.beam_center_1.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_center_1 = self.beam_center_1.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        center_1 = [posBase_center_1, posFrame_center_1, posOutput_center_1, rate_center_1]

        posBase_center_2 = self.beam_center_2.rigidBase.RigidBaseMO.position.value[:].tolist()
        posFrame_center_2 = self.beam_center_2.rigidBase.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_center_2 = self.beam_center_2.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_center_2 = self.beam_center_2.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        center_2 = [posBase_center_2, posFrame_center_2, posOutput_center_2, rate_center_2]

        posBase_FR = self.beam_legFR.rigidBase.RigidBaseMO.position.value[:].tolist()
        posFrame_FR = self.beam_legFR.rigidBase.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_FR = self.beam_legFR.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_FR = self.beam_legFR.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        FR = [posBase_FR, posFrame_FR, posOutput_FR, rate_FR]

        posBase_FL = self.beam_legFL.rigidBase.RigidBaseMO.position.value[:].tolist()
        posFrame_FL = self.beam_legFL.rigidBase.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_FL = self.beam_legFL.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_FL = self.beam_legFL.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        FL = [posBase_FL, posFrame_FL, posOutput_FL, rate_FL]

        posBase_BR = self.beam_legBR.rigidBase.RigidBaseMO.position.value[:].tolist()
        posFrame_BR = self.beam_legBR.rigidBase.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_BR = self.beam_legBR.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_BR = self.beam_legBR.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        BR = [posBase_BR, posFrame_BR, posOutput_BR, rate_BR]

        posBase_BL = self.beam_legBL.rigidBase.RigidBaseMO.position.value[:].tolist()
        posFrame_BL = self.beam_legBL.rigidBase.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_BL = self.beam_legBL.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_BL = self.beam_legBL.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        BL = [posBase_BL, posFrame_BL, posOutput_BL, rate_BL]

        posMultigait = self.multigait.MechanicalObject.position.value[:].tolist()
        self.multigait.modelCollis.collisMO.position.value[:].tolist()

        return [center_1, center_2, FR, FL, BR, BL, posMultigait]

    def setPos(self, pos):
        center_1, center_2, FR, FL, BR, BL, posMultigait = pos
        posBase_center_1, posFrame_center_1, posOutput_center_1, rate_center_1 = center_1
        posBase_center_2, posFrame_center_2, posOutput_center_2, rate_center_2 = center_2
        posBase_FR, posFrame_FR, posOutput_FR, rate_FR = FR
        posBase_FL, posFrame_FL, posOutput_FL, rate_FL = FL
        posBase_BR, posFrame_BR, posOutput_BR, rate_BR = BR
        posBase_BL, posFrame_BL, posOutput_BL, rate_BL = BL

        self.beam_center_1.rigidBase.RigidBaseMO.position.value = np.array(posBase_center_1)
        self.beam_center_2.rigidBase.RigidBaseMO.position.value = np.array(posBase_center_2)
        self.beam_legFR.rigidBase.RigidBaseMO.position.value = np.array(posBase_FR)
        self.beam_legFL.rigidBase.RigidBaseMO.position.value = np.array(posBase_FL)
        self.beam_legBR.rigidBase.RigidBaseMO.position.value = np.array(posBase_BR)
        self.beam_legBL.rigidBase.RigidBaseMO.position.value = np.array(posBase_BL)

        self.beam_center_1.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_center_1)
        self.beam_center_1.rigidBase.MappedFrames.FramesMO.position.value = np.array(posFrame_center_1)
        self.beam_center_1.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_center_1)

        self.beam_center_2.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_center_2)
        self.beam_center_2.rigidBase.MappedFrames.FramesMO.position.value = np.array(posFrame_center_2)
        self.beam_center_2.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_center_2)

        self.beam_legFR.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_FR)
        self.beam_legFR.rigidBase.MappedFrames.FramesMO.position.value = np.array(posFrame_FR)
        self.beam_legFR.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_FR)

        self.beam_legFL.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_FL)
        self.beam_legFL.rigidBase.MappedFrames.FramesMO.position.value = np.array(posFrame_FL)
        self.beam_legFL.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_FL)

        self.beam_legBR.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_BR)
        self.beam_legBR.rigidBase.MappedFrames.FramesMO.position.value = np.array(posFrame_BR)
        self.beam_legBR.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_BR)

        self.beam_legBL.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_BL)
        self.beam_legBL.rigidBase.MappedFrames.FramesMO.position.value = np.array(posFrame_BL)
        self.beam_legBL.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_BL)

        self.beam_center_1.rigidBase.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_center_2.rigidBase.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_legFR.rigidBase.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_legFL.rigidBase.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_legBR.rigidBase.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_legBL.rigidBase.MappedFrames.DiscreteCosseratMapping.init()

        self.multigait.MechanicalObject.position.value = np.array(posMultigait)

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


class AbstractMultigait():
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
        tot_length_leg= self.multigait_config["tot_length_leg"]
        tot_length_center = self.multigait_config["tot_length_center"]
        self.max_flex_center = self.multigait_config["max_flex_center"]
        self.max_flex_leg = self.multigait_config["max_flex_leg"]


        nbSectionS_leg= self.cosserat_config["nbSectionS_leg"]
        nbFramesF_leg= self.cosserat_config["nbFramesF_leg"]
        nbSectionS_center= self.cosserat_config["nbSectionS_center"]
        nbFramesF_center= self.cosserat_config["nbFramesF_center"]

        self.config_leg = {'init_pos': init_pos, 'nbSectionS': nbSectionS_leg,
                         'tot_length': tot_length_leg,
                        'nbFramesF': nbFramesF_leg, }

        self.config_center = {'init_pos': init_pos, 'nbSectionS': nbSectionS_center, 'tot_length': tot_length_center,
                        'nbFramesF': nbFramesF_center}

        self.n_cube_center = self.config_center['nbFramesF']
        self.n_cube_leg = self.config_leg['nbFramesF']

    def onEnd(self, rootNode, radius = 0.5):
        print(">>  Init Multigait...")
        self.multigait= rootNode.addChild('Multigait')

        self.config_center['init_pos'][0] = self.config_center['init_pos'][0]+ self.config_center['tot_length']/2
        self.config_center['tot_length'] = self.config_center['tot_length']/2

        p_attach =  9.281406880416528
        init_or = -2.414208305597635
        ang_g, ang_d = 25 + init_or, -25 - init_or
        quat_g, quat_d = Quaternion(axis=[0, 0, 1], degrees=ang_g), Quaternion(axis=[0, 0, 1], degrees=ang_d)
        v_quat_g, v_quat_d, s_quat_g, s_quat_d = quat_g.vector, quat_d.vector, quat_g.scalar, quat_d.scalar
        orient = [[v_quat_g[0], v_quat_g[1], v_quat_g[2], s_quat_g], [v_quat_d[0], v_quat_d[1], v_quat_d[2], s_quat_d]]


        lastFrame = {"orient": orient, "index": 2, "dist": p_attach}
        self.beam_center_1 = cosserat(rootNode, self.config_center, name = "beam_center_1", orientation = [0,0, 0, 1], radius = radius, last_frame=lastFrame, youngModulus=3169.3157527277867)
        self.beam_center_2 = cosserat(rootNode, self.config_center, name = "beam_center_2", orientation = [0,0, 1, 0], radius = radius, last_frame=lastFrame, youngModulus=3169.3157527277867)
        rigidBase_11, rigidBase_12 = self.beam_center_1.lastFrame_0, self.beam_center_1.lastFrame_1
        rigidBase_21, rigidBase_22 = self.beam_center_2.lastFrame_0, self.beam_center_2.lastFrame_1

        self.beam_legFL = cosserat(rootNode, self.config_leg, name = "beam_legFL", radius = radius, rigidBase=rigidBase_11, youngModulus=2196.145718695166)
        self.beam_legFR = cosserat(rootNode, self.config_leg, name = "beam_legFR", radius = radius, rigidBase=rigidBase_12, youngModulus=2196.145718695166)
        self.beam_legBL = cosserat(rootNode, self.config_leg, name = "beam_legBL", radius = radius, rigidBase=rigidBase_22, youngModulus=2196.145718695166)
        self.beam_legBR = cosserat(rootNode, self.config_leg, name = "beam_legBR", radius = radius, rigidBase=rigidBase_21, youngModulus=2196.145718695166)

        pos_cent = [[0., 0., 0., 0., 0., 0., 1.], [self.config_center['tot_length'], 0., 0., 0., 0., 0., 1.], [2*self.config_center['tot_length'], 0., 0., 0., 0., 0., 1.]]
        nb_point_leg = self.config_leg['nbFramesF']
        pos_relative_leg = [[(1+i)*self.config_leg['tot_length']/nb_point_leg, 0, 0] for i in range(nb_point_leg)]

        base_FL = rigidBase_11.RigidBaseMO.position.value[0]
        quat_FL = base_FL[3:]
        q_FL = Quaternion([quat_FL[3], quat_FL[0], quat_FL[1], quat_FL[2]])
        pos_FL = []
        for i in range(nb_point_leg):
            rotate_vec = q_FL.rotate(pos_relative_leg[i])
            new_pos = [self.config_center['init_pos'][0] + p_attach+ rotate_vec[0],
                       self.config_center['init_pos'][1] + rotate_vec[1],
                       self.config_center['init_pos'][2]+ rotate_vec[2]]

            pos_FL.append(new_pos + quat_FL.tolist())

        base_FR = rigidBase_12.RigidBaseMO.position.value[0]
        quat_FR = base_FR[3:]
        q_FR = Quaternion([quat_FR[3], quat_FR[0], quat_FR[1], quat_FR[2]])
        pos_FR = []
        for i in range(nb_point_leg):
            rotate_vec = q_FR.rotate(pos_relative_leg[i])
            new_pos = [self.config_center['init_pos'][0]+ p_attach + rotate_vec[0],
                       self.config_center['init_pos'][1] + rotate_vec[1],
                       self.config_center['init_pos'][2] + rotate_vec[2]]

            pos_FR.append(new_pos + quat_FR.tolist())

        base_BL = rigidBase_22.RigidBaseMO.position.value[0]
        quat_BL = base_BL[3:]
        q_BL = Quaternion([quat_BL[3], quat_BL[0], quat_BL[1], quat_BL[2]])
        pos_BL = []
        for i in range(nb_point_leg):
            rotate_vec = q_BL.rotate(pos_relative_leg[i])
            new_pos = [-rotate_vec[0]+self.config_center['init_pos'][0]- p_attach,
                       -rotate_vec[1],
                       rotate_vec[2]]
            pos_BL.append(new_pos + quat_BL.tolist())

        base_BR = rigidBase_21.RigidBaseMO.position.value[0]
        quat_BR = base_BR[3:]
        q_BR = Quaternion([quat_BR[3], quat_BR[0], quat_BR[1], quat_BR[2]])
        pos_BR = []
        for i in range(nb_point_leg):
            rotate_vec = q_BR.rotate(pos_relative_leg[i])
            new_pos = [-rotate_vec[0]+self.config_center['init_pos'][0]- p_attach,
                       -rotate_vec[1],
                       rotate_vec[2]]

            pos_BR.append(new_pos + quat_BR.tolist())

        pos_base_legs = [[self.config_center['tot_length']-p_attach, 0., 0., 0., 0., 0., 1.], [self.config_center['tot_length']+p_attach, 0., 0., 0., 0., 0., 1.]]
        positions = pos_cent + pos_FR + pos_FL + pos_BR + pos_BL + pos_base_legs
        edges = [[0, 15],  [15, 1], [1, 16], [16, 2],
                 [16, 3], [3, 4], [4, 5],
                 [16, 6], [6, 7], [7, 8],
                 [11, 10], [10, 9], [9, 15],
                 [14, 13], [13, 12], [12, 15]]


        self._addMultigait(rootNode, positions, edges, collisionGroup=0)



    def _addMultigait(self, node, positions, edges, collisionGroup=0):
        barycentre = [ -76.72398876, 53.71639887,  -3.17560985]
        barycentre[0] = barycentre[0] + self.config_center['init_pos'][0]
        lengthY_center, lengthY_leg, lengthZ = 30, 15, 6

        self.multigait.addObject('EulerImplicitSolver')
        self.multigait.addObject('SparseLDLSolver', template = 'CompressedRowSparseMatrixd', name='solver')
        self.multigait.addObject('EdgeSetTopologyContainer', position=positions, edges=edges)
        self.multigait.addObject('MechanicalObject', template='Rigid3', position=[x for x in positions])
        interpolation_legs = self.multigait.addObject('BeamInterpolation', name = 'BeamInterpolation_legs',lengthY=lengthY_leg, lengthZ=lengthZ, defaultYoungModulus=2196.145718695166, straight=False, crossSectionShape = 'rectangular',
                                    edgeList = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        interpolation_center = self.multigait.addObject('BeamInterpolation', name = 'BeamInterpolation_center', lengthY=lengthY_center, lengthZ=lengthZ, defaultYoungModulus=3169.3157527277867, straight=False, crossSectionShape = 'rectangular',
                                    edgeList = [0, 1, 2, 3])
        self.multigait.addObject('AdaptiveBeamForceFieldAndMass', name='AdaptiveBeamForceFieldAndMass_legs', massDensity=  9.037448972549561e-06, interpolation = interpolation_legs.getLinkPath())
        self.multigait.addObject('AdaptiveBeamForceFieldAndMass', name='AdaptiveBeamForceFieldAndMass_center', massDensity= 8.016637463826399e-06, interpolation = interpolation_center.getLinkPath())

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

        posBase_FR = self.beam_legFR.lastFrame_1.RigidBaseMO.position.value[:].tolist()
        posFrame_FR = self.beam_legFR.lastFrame_1.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_FR = self.beam_legFR.lastFrame_1.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_FR = self.beam_legFR.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        FR = [posBase_FR, posFrame_FR, posOutput_FR, rate_FR]

        posBase_FL = self.beam_legFL.lastFrame_0.RigidBaseMO.position.value[:].tolist()
        posFrame_FL = self.beam_legFL.lastFrame_0.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_FL = self.beam_legFL.lastFrame_0.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_FL = self.beam_legFL.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        FL = [posBase_FL, posFrame_FL, posOutput_FL, rate_FL]

        posBase_BR = self.beam_legBR.lastFrame_0.RigidBaseMO.position.value[:].tolist()
        posFrame_BR = self.beam_legBR.lastFrame_0.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_BR = self.beam_legBR.lastFrame_0.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_BR = self.beam_legBR.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        BR = [posBase_BR, posFrame_BR, posOutput_BR, rate_BR]

        posBase_BL = self.beam_legBL.lastFrame_1.RigidBaseMO.position.value[:].tolist()
        posFrame_BL = self.beam_legBL.lastFrame_1.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput_BL = self.beam_legBL.lastFrame_1.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate_BL = self.beam_legBL.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        BL = [posBase_BL, posFrame_BL, posOutput_BL, rate_BL]

        posMultigait = self.multigait.MechanicalObject.position.value[:].tolist()

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
        self.beam_legFR.lastFrame_1.RigidBaseMO.position.value = np.array(posBase_FR)
        self.beam_legFL.lastFrame_0.RigidBaseMO.position.value = np.array(posBase_FL)
        self.beam_legBR.lastFrame_0.RigidBaseMO.position.value = np.array(posBase_BR)
        self.beam_legBL.lastFrame_1.RigidBaseMO.position.value = np.array(posBase_BL)

        self.beam_center_1.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_center_1)
        self.beam_center_1.rigidBase.MappedFrames.FramesMO.position.value = np.array(posFrame_center_1)
        self.beam_center_1.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_center_1)

        self.beam_center_2.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_center_2)
        self.beam_center_2.rigidBase.MappedFrames.FramesMO.position.value = np.array(posFrame_center_2)
        self.beam_center_2.rigidBase.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_center_2)

        self.beam_legFR.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_FR)
        self.beam_legFR.lastFrame_1.MappedFrames.FramesMO.position.value = np.array(posFrame_FR)
        self.beam_legFR.lastFrame_1.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_FR)

        self.beam_legFL.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_FL)
        self.beam_legFL.lastFrame_0.MappedFrames.FramesMO.position.value = np.array(posFrame_FL)
        self.beam_legFL.lastFrame_0.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_FL)

        self.beam_legBR.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_BR)
        self.beam_legBR.lastFrame_0.MappedFrames.FramesMO.position.value = np.array(posFrame_BR)
        self.beam_legBR.lastFrame_0.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_BR)

        self.beam_legBL.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate_BL)
        self.beam_legBL.lastFrame_1.MappedFrames.FramesMO.position.value = np.array(posFrame_BL)
        self.beam_legBL.lastFrame_1.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput_BL)

        self.beam_center_1.rigidBase.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_center_2.rigidBase.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_legFR.lastFrame_1.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_legFL.lastFrame_0.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_legBR.lastFrame_0.MappedFrames.DiscreteCosseratMapping.init()
        self.beam_legBL.lastFrame_1.MappedFrames.DiscreteCosseratMapping.init()

        self.multigait.MechanicalObject.position.value = np.array(posMultigait)

# -*- coding: utf-8 -*-
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

import Sofa
import json
import numpy as np
from splib3.animation import AnimationManagerController
import os
from pyquaternion import Quaternion
from common.controller_with_proxy import ControllerWithProxy

VISUALISATION = False

pathSceneFile = os.path.dirname(os.path.abspath(__file__))
pathMesh = os.path.dirname(os.path.abspath(__file__))+'/Mesh/'
# Units: mm, kg, s.     Pressure in kPa = k (kg/(m.s^2)) = k (g/(mm.s^2) =  kg/(mm.s^2)

##########################################
# Reduced Basis Definition           #####
##########################################
modesRobot = pathSceneFile + "/ROM_data/modesQuadrupedWellConverged.txt"
nbModes = 63
modesPosition = [0 for i in range(nbModes)]

########################################################################
# Reduced Integration Domain for the PDMS membrane layer           #####
########################################################################
RIDMembraneFile = pathSceneFile + "/ROM_data/reducedIntegrationDomain_quadrupedMembraneWellConvergedNG005.txt"
weightsMembraneFile = pathSceneFile + "/ROM_data/weights_quadrupedMembraneWellConvergedNG005.txt"

#######################################################################
# Reduced Integration Domain for the main silicone body           #####
#######################################################################
RIDFile = pathSceneFile + '/ROM_data/reducedIntegrationDomain_quadrupedBodyWellConvergedNG003.txt'
weightsFile = pathSceneFile + '/ROM_data/weights_quadrupedBodyWellConvergedNG003.txt'

##############################################################
# Reduced Integration Domain in terms of nodes           #####
##############################################################
listActiveNodesFile = pathSceneFile + '/ROM_data/listActiveNodes_quadrupedBodyMembraneWellConvergedNG003and005.txt'

##########################################
# Reduced Order Booleans             #####
##########################################
performECSWBoolBody = True
performECSWBoolMembrane = True
performECSWBoolMappedMatrix = True
prepareECSWBool = False


class GetInfos(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.effectors = kwargs["effectors"]
        self.points = kwargs["points"]
        self.actuators = kwargs["actuators"]
        self.step_inverse = kwargs["step_inverse"]
        self.infos = None

        self.id = 0
        self.valuesToSave = [[], [], [], [], [], [], []]
        self.current_time = 0


    def _getInfos(self):
        actuation = [float(actuator.volumeGrowth.value) for actuator in self.actuators]
        effectorsPos = self.effectors.MechanicalObject.position.value[:].tolist()
        points = self.points.MechanicalObject.position.value[[12, 8, 20, 16]].tolist()

        self.infos = {"actuation": actuation, "effectorsPos": effectorsPos, "points": points}

    def onAnimateEndEvent(self, event):
        self._getInfos()

        self.id += 1
        if self.id%self.step_inverse==0:
            print(">> Dump data inverse.")
            self.current_time+=1
            for i, actuator in enumerate(self.actuators):
                self.valuesToSave[i].append(actuator.volumeGrowth.value)
            self.valuesToSave[-2].append(self.current_time)
            self.valuesToSave[-1].append(0)

            with open("./inverse_controlled_volumeGrowth_and_reward.txt", 'w') as outfile:
                json.dump(self.valuesToSave, outfile)

class GetReward(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.effectors = kwargs["effectors"]
    def update(self):
        self.init_pos = np.array([float(p) for p in self.effectors.MechanicalObject.position.value[0, :3]])

    def getReward(self):
        current_pos = self.effectors.MechanicalObject.position.value[0, :3]
        return current_pos[0]-self.init_pos[0], False

class Actuate(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.actuators = kwargs["actuators"]
        self.values = None
        self.translation = np.array(kwargs["translation"])

        self.current_time = 0
        self.valuesToSave = [[], [], [], [], [], [], []]

    def getInfos(self):
        barycentre = self.root.solverNode.reducedModel.model.Effectors.MechanicalObject.position.value[:,:3].mean(axis = 0)
        barycentre -= self.translation
        return {"barycentre" :barycentre.tolist()}

    def _setValue(self, values):
        self.values = values

    def save_actuators_state(self):
        pressures = []
        for actuator in self.actuators:
            pressures.append(actuator.pressure.value)
        np.savetxt(pathSceneFile + "/CavityPressure.txt", pressures)
        print("[INFO]  >>  Save pressure at "+pathSceneFile + "/CavityPressure.txt")

    def onAnimateBeginEvent(self, event):
        if not(self.values is None):
            for actuator, value in zip(self.actuators, self.values):
                actuator.value.value = [value]

    def onAnimateEndEvent(self, event):
        self.current_time += 1
        if self.current_time < 1080: #nb_action * scale = 18 * 60
            print(">> Dump direct data.")
            reward = self.root.getReward.getReward()
            for i, actuator in enumerate(self.actuators):
                self.valuesToSave[i].append(actuator.volumeGrowth.value)
            self.valuesToSave[-2].append(self.current_time)
            self.valuesToSave[-1].append(reward)

            with open("./direct_controlled_volumeGrowth_and_reward.txt", 'w') as outfile:
                json.dump(self.valuesToSave, outfile)

    def action_rescaling(self, points_sofagym, points_inverse):
        orientation = []
        for point_sofagym, point_inverse in zip(points_sofagym, points_inverse):
            q_sofagym = [point_sofagym[-1], point_sofagym[-4],
                         point_sofagym[-3], point_sofagym[-2]]
            q_sofagym = Quaternion(*q_sofagym)
            angle_sofagym = q_sofagym.degrees

            q_inverse = [point_inverse[-1], point_inverse[-4],
                         point_inverse[-3], point_inverse[-2]]
            q_inverse = Quaternion(*q_inverse)
            angle_inverse = q_inverse.degrees

            orientation.append(abs(angle_inverse/angle_sofagym))

        return np.array([max(orientation)] + orientation)


class MoveGoal(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.goals = kwargs["goals"]
        self.inverse_model = kwargs["inverse_model"]
        self.waitingtime = kwargs["waitingtime"]
        self.inverse = kwargs["inverse"]
        self.translation = np.array(kwargs["translation"])

        self.goal_pos = None
        self.idx_goal = kwargs["idx_goal"]

        self.idx_pos = 0
        self.pos = None

    def update_goal(self, goal_pos):
        self.goal_pos = goal_pos
        self.idx_goal = 0

    def update_pos(self, pos):
        self.pos = pos
        self.idx_pos = 0

    def getCorrectedPID(self, position, Kp):
        effectorsPos_direct = self.root.solverNode.reducedModel.model.Effectors.MechanicalObject.position.value[:, :3]
        return position + Kp*(position+self.translation-effectorsPos_direct)

    def onAnimateBeginEvent(self, event):
        # if not self.inverse:
        #     print("Effectors:",  self.root.solverNode.reducedModel.model.Effectors.MechanicalObject.position.value[:, :3].mean(axis = 0))
        if not (self.goal_pos is None) and self.idx_goal < len(self.goal_pos):
            pos = self.goal_pos[self.idx_goal] + self.translation
            if self.inverse:
                self.goals.effectorGoal.value = pos
            else:
                self.goals.position.value = pos
            self.idx_goal+=1

        if not (self.pos is None) and self.idx_pos < len(self.pos) and not self.inverse:
            pos = self.pos[self.idx_pos]
            self.inverse_model.position.value = pos
            self.idx_pos+=1

def createScene(rootNode, config={"source": [220, -500, 100],
                                  "target": [220, 0, 0],
                                  "goalPos": [0, 0, 0],
                                  "inverse": False,
                                  "waitingtime": 0,
                                  "readTime": 0.0,
                                  "idx":0}):


    rootNode.addObject('RequiredPlugin', name='SoftRobots', pluginName='SoftRobots')
    rootNode.addObject('RequiredPlugin', name='SofaPython', pluginName='SofaPython3')
    rootNode.addObject('RequiredPlugin', name='ModelOrderReduction', pluginName='ModelOrderReduction')
    rootNode.addObject('RequiredPlugin', name='SofaOpenglVisual')
    rootNode.addObject('RequiredPlugin', name="SofaSparseSolver")
    rootNode.addObject('RequiredPlugin', name="SofaConstraint")
    rootNode.addObject('RequiredPlugin', name="SofaEngine")
    rootNode.addObject('RequiredPlugin', name="SofaImplicitOdeSolver")
    rootNode.addObject('RequiredPlugin', name="SofaLoader")
    rootNode.addObject('RequiredPlugin', name="SofaMeshCollision")
    rootNode.addObject('RequiredPlugin', name ="SofaGeneralLoader")
    rootNode.addObject('RequiredPlugin', name='SoftRobots.Inverse')
    rootNode.addObject('RequiredPlugin', name='SofaMiscMapping')
    rootNode.addObject('RequiredPlugin', name='SofaExporter')
    rootNode.dt.value = 0.01

    rootNode.gravity.value = [0, 0, -9810]
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                   'hideBoundingCollisionModels hideForceFields '
                                                   'showInteractionForceFields hideWireframe')

    rootNode.addObject("DefaultVisualManagerLoop")
    rootNode.addObject('FreeMotionAnimationLoop')
    if config["inverse"]:
        rootNode.addObject('QPInverseProblemSolver', allowSliding = True, responseFriction=0.7,
                             maxIterations=3000, tolerance=1e-5)#, epsilon = 0.001)
    else:
        rootNode.addObject('GenericConstraintSolver', printLog=False, tolerance=1e-6, maxIterations=3000)
    rootNode.addObject('DefaultPipeline', verbose=0)
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('DefaultContactManager', response="FrictionContactConstraint", responseParams="mu=0.7")
    rootNode.addObject('LocalMinDistance', name="Proximity", alarmDistance=5, contactDistance=0.4, angleCone=0.01)

    rootNode.addObject('BackgroundSetting', color=[0, 0, 0, 1])

    solverNode = rootNode.addChild('solverNode')

    solverNode.addObject('EulerImplicitSolver', name='odesolver',firstOrder=False, rayleighStiffness=0.1,
                             rayleighMass=0.1, printLog=False)
    solverNode.addObject('SparseLDLSolver', name="preconditioner", template="CompressedRowSparseMatrixd")
    solverNode.addObject('GenericConstraintCorrection', solverName='preconditioner')
    solverNode.addObject('MechanicalMatrixMapperMOR', template='Vec1d,Vec1d', object1='@./reducedModel/alpha',
                             object2='@./reducedModel/alpha', nodeToParse='@./reducedModel/model',
                             performECSW=performECSWBoolMappedMatrix, listActiveNodesPath=listActiveNodesFile,
                             timeInvariantMapping1=True, timeInvariantMapping2=True, saveReducedMass=False,
                             usePrecomputedMass=False, precomputedMassPath='ROM_data/quadrupedMass_reduced63modes.txt',
                             printLog=False)

    ##########################################
    # FEM Reduced Model                      #
    ##########################################
    reducedModel = solverNode.addChild('reducedModel')
    reducedModel.addObject('MechanicalObject', template='Vec1d', name='alpha', position=modesPosition, printLog=False)
    ##########################################
    # FEM Model                              #
    ##########################################
    youngModulus_model, youngModulus_sub = 70, 5000 #70, 5000
    poissonRatio_model, poissonRatio_sub = 0.05, 0.49 #0.05, 0.49

    model = reducedModel.addChild('model')
    model.addObject('MeshVTKLoader', name='loader', filename=pathMesh+'full_quadriped_fine.vtk') #full_quadriped_SMALL.vtk
    model.addObject('TetrahedronSetTopologyContainer', src='@loader')
    model.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false', showIndicesScale=4e-5,
                    rx=0, printLog=False)
    model.addObject('ModelOrderReductionMapping', input='@../alpha', output='@./tetras', modesPath=modesRobot,
                    printLog=False, mapMatrices=0)
    model.addObject('UniformMass', name='quadrupedMass', totalMass=0.035, printLog=False)
    model.addObject('HyperReducedTetrahedronFEMForceField', template='Vec3d',
                    name='Append_HyperReducedFF_QuadrupedWellConverged_'+str(nbModes)+'modes', method='large',
                    poissonRatio=poissonRatio_model,  youngModulus=youngModulus_model, prepareECSW=prepareECSWBool,
                    performECSW=performECSWBoolBody, nbModes=str(nbModes), modesPath=modesRobot, RIDPath=RIDFile,
                    weightsPath=weightsFile, nbTrainingSet=93, periodSaveGIE=50,printLog=False)
    model.addObject('BoxROI', name='boxROISubTopo', box=[0, 0, 0, 150, -100, 1], drawBoxes=False)
    model.addObject('BoxROI', name='membraneROISubTopo', box=[0, 0, -0.1, 150, -100, 0.1], computeTetrahedra=False,
                    drawBoxes=False)

    ##########################################
    # Sub topology                           #
    ##########################################
    modelSubTopo = model.addChild('modelSubTopo')
    modelSubTopo.addObject('TriangleSetTopologyContainer', position='@membraneROISubTopo.pointsInROI',
                           triangles="@membraneROISubTopo.trianglesInROI", name='container')
    modelSubTopo.addObject('HyperReducedTriangleFEMForceField', template='Vec3d', name='Append_subTopoFEM',
                           method='large', poissonRatio=poissonRatio_sub,  youngModulus=youngModulus_sub, prepareECSW=prepareECSWBool,
                           performECSW=performECSWBoolMembrane, nbModes=str(nbModes), modesPath=modesRobot,
                           RIDPath=RIDMembraneFile, weightsPath=weightsMembraneFile, nbTrainingSet=93,
                           periodSaveGIE=50, printLog=False)


    ##########################################
    # Constraint                             #
    ##########################################
    if config["inverse"]:
        try:
            init_pressure = np.loadtxt(pathSceneFile + "/CavityPressure.txt")
        except:
            print("[WARNING]  >> No init_pressure. (file not found: "+pathSceneFile + "/CavityPressure.txt)")
            init_pressure = [0 for _ in range(5)]
    centerCavity = model.addChild('centerCavity')
    centerCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Center-cavity_finer.stl')
    centerCavity.addObject('MeshTopology', src='@loader', name='topo')
    centerCavity.addObject('MechanicalObject', name='centerCavity')
    if config["inverse"]:
        centerCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',
                         triangles='@topo.triangles', drawPressure=0, drawScale=0.0002, initPressure = init_pressure[0], minPressure= 0)#0.0001, maxPressure = 5, maxPressureVariation = 0.3)
    else:
        centerCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                               value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                               valueType="volumeGrowth") # valueType="pressure")
    centerCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    rearLeftCavity = model.addChild('rearLeftCavity')
    rearLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Left-cavity_finer.stl')
    rearLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    rearLeftCavity.addObject('MechanicalObject', name='rearLeftCavity')
    if config["inverse"]:
        rearLeftCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',triangles='@topo.triangles', drawPressure=0,
                                 drawScale=0.0002, initPressure = init_pressure[1], minPressure= 0) #0.0001, maxPressure = 3, maxPressureVariation = 0.3)
    else:
        rearLeftCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                               value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                               valueType="volumeGrowth") # valueType="pressure")
    rearLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')

    rearRightCavity = model.addChild('rearRightCavity')
    rearRightCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Right-cavity_finer.stl')
    rearRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    rearRightCavity.addObject('MechanicalObject', name='rearRightCavity')
    if config["inverse"]:
        rearRightCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',
                                  triangles='@topo.triangles', drawPressure=0, drawScale=0.0002, initPressure = init_pressure[2], minPressure= 0) #0.0001, maxPressure = 3, maxPressureVariation =  0.3)
    else:
        rearRightCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                               value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                               valueType="volumeGrowth") # valueType="pressure")
    rearRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    frontLeftCavity = model.addChild('frontLeftCavity')
    frontLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Front-Left-cavity_finer.stl')
    frontLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    frontLeftCavity.addObject('MechanicalObject', name='frontLeftCavity')
    if config["inverse"]:
        frontLeftCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',
                                      triangles='@topo.triangles', drawPressure=0, drawScale=0.0002, initPressure = init_pressure[3], minPressure= 0) #0.0001, maxPressure = 3, maxPressureVariation =  0.3)
    else:
        frontLeftCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                               value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                               valueType="volumeGrowth") # valueType="pressure")
    frontLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')

    frontRightCavity = model.addChild('frontRightCavity')
    frontRightCavity.addObject('MeshSTLLoader', name='loader',
                               filename=pathMesh+'quadriped_Front-Right-cavity_finer.stl')
    frontRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    frontRightCavity.addObject('MechanicalObject', name='frontRightCavity')
    if config["inverse"]:
        frontRightCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',
            triangles='@topo.triangles', drawPressure=0, drawScale=0.0002, initPressure = init_pressure[4], minPressure= 0) #0.0001, maxPressure = 3, maxPressureVariation =  0.3)
    else:
        frontRightCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                               value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                               valueType="volumeGrowth") # valueType="pressure")
    frontRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    modelCollis = model.addChild('modelCollis')
    modelCollis.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_collision.stl',
                              rotation=[0, 0, 0], translation=[0, 0, 0])
    modelCollis.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    modelCollis.addObject('MechanicalObject', name='collisMO', template='Vec3d')
    modelCollis.addObject('TriangleCollisionModel', group=0)
    modelCollis.addObject('LineCollisionModel', group=0)
    modelCollis.addObject('PointCollisionModel', group=0)
    modelCollis.addObject('BarycentricMapping')

    ##########################################
    # Visualization                          #
    ##########################################
    modelVisu = model.addChild('visu')
    modelVisu.addObject('MeshSTLLoader', name='loader', filename=pathMesh+"quadriped_collision.stl")
    modelVisu.addObject('OglModel', src='@loader', template='Vec3d', color=[0.7, 0.7, 0.7, 0.6])
    modelVisu.addObject('BarycentricMapping')

    planeNode = rootNode.addChild('Plane')
    planeNode.addObject('MeshOBJLoader', name='loader', filename="mesh/floorFlat.obj", triangulate="true")
    planeNode.addObject('MeshTopology', src="@loader")
    planeNode.addObject('MechanicalObject', src="@loader", rotation=[90, 0, 0], translation=[250, 35, -1], scale=15)

    planeNode.addObject('OglModel', name="Visual", src="@loader", color=[1, 1, 1, 0.5], rotation=[90, 0, 0],
                            translation=[250, 35, -1], scale=15)
    planeNode.addObject('TriangleCollisionModel', simulated=0, moving=0, group=1)
    planeNode.addObject('LineCollisionModel', simulated=0, moving=0, group=1)
    planeNode.addObject('PointCollisionModel', simulated=0, moving=0, group=1)
    planeNode.addObject('UncoupledConstraintCorrection')
    planeNode.addObject('EulerImplicitSolver', name='odesolver')
    planeNode.addObject('CGLinearSolver', name='Solver', iterations= 500, tolerance=1e-5, threshold=1e-5)

    rootNode.addObject(AnimationManagerController(rootNode, name="AnimationManager"))

    source = config["source"]
    target = config["target"]
    rootNode.addObject("LightManager")
    spotLoc = [0, 0, 1000]
    rootNode.addObject("SpotLight", position=spotLoc, direction=[0, 0.0, -1.0])
    rootNode.addObject("InteractiveCamera", name="camera", position=source, lookAt=target, zFar=500)

    # init_pos = np.array([[ 2.94999910e+01, -2.86148631e-05, -1.85017111e+01],
    #                      [ 4.91666627e+01, -1.68939898e-05, -1.84982822e+01],
    #                      [ 5.89999979e+01, -1.26175857e-05, -1.84999169e+01],
    #                      [ 9.83332133e+00, -2.04778988e-05, -1.84981941e+01],
    #                      [-1.51512943e-05,  2.93742937e-06, -1.84999396e+01],
    #                      [ 5.89999802e+01,  1.07371094e+01, -1.84997799e+01],
    #                      [ 7.07795716e+01,  1.50245286e+01, -1.85019394e+01],
    #                      [ 8.25591690e+01,  1.93119498e+01, -1.85006680e+01],
    #                      [ 9.43387674e+01,  2.35993768e+01, -1.84994903e+01],
    #                      [ 5.89999963e+01, -1.07371347e+01, -1.85000975e+01],
    #                      [ 7.07795823e+01, -1.50245468e+01, -1.85014149e+01],
    #                      [ 8.25591748e+01, -1.93119628e+01, -1.85006876e+01],
    #                      [ 9.43387746e+01, -2.35993830e+01, -1.84993904e+01],
    #                      [ 5.84709069e-05,  1.07371697e+01, -1.84999545e+01],
    #                      [-1.17795610e+01,  1.50245744e+01, -1.85022369e+01],
    #                      [-2.35592192e+01,  1.93118342e+01, -1.85000530e+01],
    #                      [-3.53388682e+01,  2.35991609e+01, -1.84985120e+01],
    #                      [ 5.61156298e-06, -1.07371417e+01, -1.84999232e+01],
    #                      [-1.17795823e+01, -1.50245660e+01, -1.85028468e+01],
    #                      [-2.35591851e+01, -1.93120014e+01, -1.85041731e+01],
    #                      [-3.53387956e+01, -2.35994636e+01, -1.84962351e+01]])
    init_pos = np.array([[29.500350135242684, -0.0015928885371324928, -18.50171103306455, -5.957860824270543e-06, -1.410096780268166e-05, -1.617815088208691e-05, 0.9999999997519677],
              [49.167031877422616, -0.0017420842050015924, -18.498282164222424, 5.680612564400278e-06, 0.0002402416678295303, 1.607264586117921e-05, 0.999999970996669],
              [59.00037025215806, -0.0013360323731174916, -18.499916976824174, 1.35495676299469e-05, -0.00017410438504077776, 2.574539029454111e-05, 0.9999999844206326],
              [9.833688096734543, -0.0001637476078494213, -18.498194190504393, -5.85528830786701e-05, -0.00023952002566699396, -5.627954007598284e-05, 0.9999999680171623],
              [0.0003303321348077449, 0.0009017508032585831, -18.499939612778153, -7.149275199620997e-05, 0.00014280897831524925, -5.060608570249165e-05, 0.9999999859667047],
              [59.00017607866042, 10.735599067069476, -18.49977989151812, 0.0001278521703467651, -0.00014336867993295158, 0.17366871372400103, 0.9848043286565571],
              [70.78012223290574, 15.023667471750546, -18.50193931572959, -2.401195477693528e-05, -0.00014061805876265413, 0.17365689988729843, 0.9848064203382922],
              [82.55998125195673, 19.311245338681136, -18.50066802395987, 2.9313176268351615e-05, 6.504337759175767e-05, 0.17364270835438259, 0.9848089304598384],
              [94.33961533767587, 23.598677270072404, -18.499490306262654, -4.5691270313425904e-06, 5.3043697233926356e-05, 0.17365410111195204, 0.9848069227521148],
              [59.00103243311104, -10.738460797476408, -18.50009764890077, -0.00014998879127972313, -8.408305040703731e-05, -0.17362226176054826, 0.9848125229821942],
              [70.78024282239396, -15.025259858892598, -18.501414934838305, -4.493314644722793e-05, -0.00021900535480814923, -0.1736333722874086, 0.9848105537684743],
              [82.5597178752086, -19.31241765553792, -18.500687498390956, 3.845179834900959e-05, 0.00015945008680130285, -0.17364143237055407, 0.9848091443672881],
              [94.33949008755822, -23.599882503405862, -18.499390404090363, -5.443267527719091e-05, -2.8439531680870995e-05, -0.1736512530447143, 0.9848074244806833],
              [0.0007317688613944463, 10.73829717502417, -18.49995453763681, -0.00017755635192780167, -1.6835095023383663e-05, 0.1736529871990501, -0.9848071044601411],
              [-11.779745567184792, 15.024868980528671, -18.50223697565891, 3.375076203533564e-05, -6.291668862093025e-05, 0.17355820006852887, -0.9848238273210845],
              [-23.560056325758026, 19.310799084818544, -18.50005316624177, -1.832328591272879e-06, 0.0002400767909200606, 0.17361030350281123, -0.984814616892661],
              [-35.340063730931476, 23.59734772066143, -18.49851185661924, -4.267886435283499e-05, -5.071005866505608e-05, 0.1736281844229451, -0.9848114915804913],
              [2.2914119991053732e-06, -10.737257661356528, -18.49992312737645, 2.3403390926317287e-05, -0.00012430280077349294, -0.17362239127744175, -0.9848125070368465],
              [-11.779566656086105, -15.024689887405454, -18.502846816999682, 4.321443137194397e-05, 9.575088623082308e-06, -0.17365443022779753, -0.9848068651624483],
              [-23.559301273401992, -19.311766954933223, -18.50417285534941, -4.665846291011694e-06, 0.0002958154017070519, -0.173613375560309, -0.9848140601484291],
              [-35.33925196425221, -23.59830763886671, -18.496235240786866, -7.576548585319785e-05, 0.0006779160875639976, -0.17361274924918976, -0.9848139787571583]])
    translation =  np.array([ 47.38832382, -53.62797805,  18.5002594 ])
    init_pos[:, :3] += translation

    if config["inverse"]:
        actuators = [centerCavity.SurfacePressureActuator,
                      rearLeftCavity.SurfacePressureActuator,
                      rearRightCavity.SurfacePressureActuator,
                      frontLeftCavity.SurfacePressureActuator,
                      frontRightCavity.SurfacePressureActuator]


        effectors = model.addChild('Effectors')
        effectors.addObject('MechanicalObject', position = init_pos[:, :3], showObject = True, showObjectScale=10, showColor=[0, 0, 1, 1])
        effectors.addObject('PositionEffector', effectorGoal= init_pos[:, :3], indices = [i for i in range(len(init_pos))], weight = 1)
        effectors.addObject('BarycentricMapping')

        points = model.addChild('Points')
        points.addObject('MechanicalObject', position = init_pos, template = "Rigid3", showObject = True, showObjectScale=10, showColor=[0, 0, 1, 1])
        points.addObject('BarycentricMapping')

        rootNode.addObject(GetInfos(name="getInfos", actuators = actuators, effectors = effectors, points = points,  step_inverse=5))
        rootNode.addObject(MoveGoal(name="moveGoal",root = rootNode,  goals = effectors.PositionEffector, waitingtime = config["waitingtime"], idx_goal = config["idx"], inverse= True, translation = np.array([ 46.6697038 , -53.629322  ,  18.50050114]),   inverse_model = None))

        try:
            readTime = config["readTime"]
            model.addObject('ReadState', name="states", filename= pathSceneFile+"/modelState", shift = readTime, printLog = "1")
            reducedModel.addObject('ReadState', name="states", filename= pathSceneFile+"/reducedModel", shift = readTime, printLog = "1")
            modelSubTopo.addObject('ReadState', name="states", filename= pathSceneFile+"/modelSubTopoState", shift = readTime, printLog = "1")
            centerCavity.addObject('ReadState', name="states", filename= pathSceneFile+"/centerCavityState", shift = readTime, printLog = "1")
            rearLeftCavity.addObject('ReadState', name="states", filename= pathSceneFile+"/rearLeftCavityState", shift = readTime, printLog = "1")
            rearRightCavity.addObject('ReadState', name="states", filename= pathSceneFile+"/rearRightCavityState", shift = readTime, printLog = "1")
            frontLeftCavity.addObject('ReadState', name="states", filename= pathSceneFile+"/frontLeftCavityState", shift = readTime, printLog = "1")
            frontRightCavity.addObject('ReadState', name="states", filename= pathSceneFile+"/frontRightCavityState", shift = readTime, printLog = "1")
            modelCollis.addObject('ReadState', name="states", filename= pathSceneFile+"/modelCollisState", shift = readTime, printLog = "1")
        except:
            print("[WARNING]  >> No read state available. Make sure filenames exist.")

    else:
        writeTime = 60*0.01
        scale = 0 #60*0.01
        w_1 = model.addObject('WriteState', name="writer", filename= pathSceneFile+"/modelState", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)#, period = scale)
        w_2 = reducedModel.addObject('WriteState', name="writer", filename= pathSceneFile+"/reducedModel", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)#, period = scale)
        w_3 = modelSubTopo.addObject('WriteState', name="writer", filename= pathSceneFile+"/modelSubTopoState", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)#, period = scale)
        w_4 = centerCavity.addObject('WriteState', name="writer", filename= pathSceneFile+"/centerCavityState", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)#, period = scale)
        w_5 = rearLeftCavity.addObject('WriteState', name="writer", filename= pathSceneFile+"/rearLeftCavityState", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)#, period = scale)
        w_6 = rearRightCavity.addObject('WriteState', name="writer", filename= pathSceneFile+"/rearRightCavityState", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)#, period = scale)
        w_7 = frontLeftCavity.addObject('WriteState', name="writer", filename= pathSceneFile+"/frontLeftCavityState", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)#, period = scale)
        w_8 = frontRightCavity.addObject('WriteState', name="writer", filename= pathSceneFile+"/frontRightCavityState", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)#, period = scale)
        w_9 = modelCollis.addObject('WriteState', name="writer", filename= pathSceneFile+"/modelCollisState", time = writeTime, printLog = "1", writeX0 = True, writeF = True, writeV = True)#, period = scale)
        writers = [w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9]


        effectors = model.addChild('Effectors')
        effectors.addObject('MechanicalObject', position = init_pos[:, :3], showObject = True, showObjectScale=10, showColor=[0, 1, 0, 1])
        effectors.addObject('BarycentricMapping')

        goals = rootNode.addChild("Goals")
        goals.addObject('MechanicalObject', position=init_pos[:, :3], showObject = True, showObjectScale=10, showColor=[0, 0, 1, 1])

        inverse_model = rootNode.addChild("inverseModel")
        inverse_model.addObject('MechanicalObject', position=init_pos[:, :3], showObject = True, showObjectScale=5, showColor=[1, 0, 0, 1])

        actuators = [centerCavity.SurfacePressureConstraint,
                      rearLeftCavity.SurfacePressureConstraint,
                      rearRightCavity.SurfacePressureConstraint,
                      frontLeftCavity.SurfacePressureConstraint,
                      frontRightCavity.SurfacePressureConstraint]


        #Controller with proxy
        config.update({"inverse": True})
        specific_actions = np.array([4, 0, 0, 2, 5, 1, 3, 4])

        noise_level = 0.1
        rootNode.addObject(ControllerWithProxy(name="ControllerWithProxy", root = rootNode, inverse_scene_path = "sofagym.env.QPMultigait.MORMultiGaitRobotScene",
                            scale = 60, env_name = "optiabstractmultigait-v0", save_rl_path = "../../../Results_benchmark/PPO_optiabstractmultigait-v0_30/best",
                            name_algo = "PPO", config = config, writers = writers, translation = translation,
                            time_register_sofagym = 7, time_register_inverse = 7, nb_step_inverse=3, init_action = [0, 0, 0, 0, 0],
                            min_bounds = [0, 0, 0, 0, 0], max_bounds = [2050, 1600, 1600, 1600, 1600],
                            factor_commande = 0.3, Kp = 0.5, specific_actions = specific_actions, name_save = "../../../Proxy/Results/Multigait_"+str(noise_level)+"/",
                            effectors = effectors, goals = goals, use_cumulative_reward_sofagym = True))


        rootNode.addObject(GetReward(name="getReward", effectors = effectors))
        rootNode.addObject(Actuate(name="actuate", actuators = actuators, root = rootNode, translation = translation))
        rootNode.addObject(MoveGoal(name="moveGoal", root = rootNode, goals = goals.MechanicalObject, waitingtime = config["waitingtime"], idx_goal = config["idx"], inverse= False, translation = np.array([ 46.6697038 , -53.629322  ,  18.50050114]), inverse_model = inverse_model.MechanicalObject))



    return rootNode

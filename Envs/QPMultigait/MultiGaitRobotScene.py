
import sys
import pathlib
import json
import numpy as np
import Sofa
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from splib3.animation import AnimationManagerController
import os


pathSceneFile = os.path.dirname(os.path.abspath(__file__))
pathMesh = os.path.dirname(os.path.abspath(__file__))+'/Mesh/'
# Units: mm, kg, s.     Pressure in kPa = k (kg/(m.s^2)) = k (g/(mm.s^2) =  kg/(mm.s^2)


def add_goal_node(root, position = [0, 0.0, 0.0], name = "Goal"):
    goal = root.addChild(name)
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject = True, showObjectScale = 2, showColor = [255, 0, 0, 255], drawMode = 1, position=position)
    return goal_mo

def changePressure(rootNode, pressure, nb_step):
    node = rootNode.model
    node.centerCavity.SurfacePressureConstraint.value.value = \
        node.centerCavity.SurfacePressureConstraint.value.value + pressure[0]/nb_step
    node.rearLeftCavity.SurfacePressureConstraint.value.value = \
        node.rearLeftCavity.SurfacePressureConstraint.value.value + pressure[1]/nb_step
    node.rearRightCavity.SurfacePressureConstraint.value.value = \
        node.rearRightCavity.SurfacePressureConstraint.value.value + pressure[2]/nb_step
    node.frontLeftCavity.SurfacePressureConstraint.value.value = \
        node.frontLeftCavity.SurfacePressureConstraint.value.value + pressure[3]/nb_step
    node.frontRightCavity.SurfacePressureConstraint.value.value = \
        node.frontRightCavity.SurfacePressureConstraint.value.value + pressure[4]/nb_step

def action_to_command(action, rootNode):
    pressure_leg, pressure_center = 1500, 2000
    node = rootNode.model

    a_center, b_center = pressure_center/2, pressure_center/2
    a_leg, b_leg = pressure_leg/2, pressure_leg/2

    goal1 = a_center*action[0]+ b_center
    goal2 = a_leg*action[1] + b_leg
    goal3 = a_leg*action[2] + b_leg
    goal4 = a_leg*action[3] + b_leg
    goal5 = a_leg*action[4] + b_leg

    old1 = float(node.centerCavity.SurfacePressureConstraint.value.value[0])
    old2 = float(node.rearLeftCavity.SurfacePressureConstraint.value.value[0])
    old3 = float(node.rearRightCavity.SurfacePressureConstraint.value.value[0])
    old4 = float(node.frontLeftCavity.SurfacePressureConstraint.value.value[0])
    old5 = float(node.frontRightCavity.SurfacePressureConstraint.value.value[0])

    incr = [goal1 - old1, goal2-old2, goal3-old3, goal4 - old4, goal5 - old5]
    return incr

class ApplyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root =  kwargs["root"]
        self.actions = kwargs["actions"]
        self.scale = kwargs["scale"]

        self.current_idx = 0
        self.already_done = 0
        self.current_incr = None

    def onAnimateBeginEvent(self, event):
        if self.already_done%self.scale == 0:
            current_action = self.actions[self.current_idx]
            self.current_incr = action_to_command(current_action, self.root)
            self.current_idx+=1
        changePressure(self.root, self.current_incr , self.scale)
        self.already_done+=1

class MoveGoal(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root =  kwargs["root"]
        self.positions = kwargs["positions"]

        self.current_idx = 0

    def onAnimateBeginEvent(self, event, translation = [55.26272867193276, -53.71608077293722, 17.499841404638524]):
        # if self.current_idx == 0:
        #     self.init_pos = self._compute_pos()[0]

        if self.current_idx < len(self.positions):
            print(">>  MOVE.")
            current_pos = np.array(self.positions[self.current_idx])
            current_pos[:,0]+= translation[0]
            current_pos[:,1]+= translation[1]
            current_pos[:,2]+= translation[2]
            self.root.Goals.GoalMO.position.value = current_pos
            self.root.model.modelCollis.Effectors.PositionEffector.effectorGoal.value = current_pos

            for i in range(len(current_pos)):
                print(i, ":", np.linalg.norm(current_pos[i]- self.root.model.modelCollis.Effectors.PositionEffector.effectorGoal.value[i]),current_pos[i], self.root.model.modelCollis.Effectors.PositionEffector.effectorGoal.value[i])


            self.current_idx += 1

    # def onAnimateEndEvent(self, event):
    #     error = self._compute_error()
    #     print(">> Error:", error)
    #     print(">> Distance:", self.init_pos - self._compute_pos()[0])
    #
    # def _compute_error(self):
    #     pos_goal = self.root.Goals.GoalMO.position.value
    #     pos_effectors = self.root.model.modelCollis.Effectors.EffectorMO.position.value
    #     return np.linalg.norm(pos_goal-pos_effectors, axis = 1).mean()
    #
    # def _compute_pos(self):
    #     pos_effectors = self.root.model.modelCollis.Effectors.EffectorMO.position.value
    #     pos_barycentre = pos_effectors.mean(axis = 0)
    #     return pos_barycentre


def createScene(rootNode, config={"source": [220, -500, 100],
                                  "target": [220, 0, 0],
                                  "goalPos": [0, 0, 0]}, mode='simu_and_visu'):

    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    rootNode.addObject('RequiredPlugin', name='SoftRobots', pluginName='SoftRobots')
    rootNode.addObject('RequiredPlugin', name='SofaPython', pluginName='SofaPython3')
    rootNode.addObject('RequiredPlugin', name='SofaOpenglVisual')
    rootNode.addObject('RequiredPlugin', name="SofaSparseSolver")
    rootNode.addObject('RequiredPlugin', name="SofaConstraint")
    rootNode.addObject('RequiredPlugin', name="SofaEngine")
    rootNode.addObject('RequiredPlugin', name="SofaImplicitOdeSolver")
    rootNode.addObject('RequiredPlugin', name="SofaLoader")
    rootNode.addObject('RequiredPlugin', name="SofaMeshCollision")
    rootNode.addObject('RequiredPlugin', name ="SofaGeneralLoader")
    rootNode.addObject('RequiredPlugin', name='SoftRobots.Inverse')
    rootNode.addObject('RequiredPlugin', name='SofaBoundaryCondition')
    rootNode.addObject('RequiredPlugin', name='SofaSimpleFem')
    rootNode.addObject('RequiredPlugin', name='SofaDeformable')
    rootNode.addObject('RequiredPlugin', name='SofaMiscFem')

    rootNode.dt.value = 0.01

    rootNode.addObject('DefaultVisualManagerLoop')

    rootNode.gravity.value = [0, 0, -9810]
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels showCollisionModels '
                                                   'hideBoundingCollisionModels hideForceFields '
                                                   'showInteractionForceFields hideWireframe')
    rootNode.addObject('BackgroundSetting', color=[0, 0, 0, 1])

    rootNode.addObject('FreeMotionAnimationLoop')
    #Inverse
    rootNode.addObject('QPInverseProblemSolver', allowSliding = True, responseFriction=0.7)
    # #Directe
    # rootNode.addObject('GenericConstraintSolver', printLog=False, tolerance=1e-6, maxIterations=500)

    rootNode.addObject('DefaultPipeline', verbose=0)
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('DefaultContactManager', response="FrictionContactConstraint", responseParams="mu=0.7")
    rootNode.addObject('LocalMinDistance', name="Proximity", alarmDistance=2, contactDistance=0.5, angleCone=0.01)
    rootNode.addObject(AnimationManagerController(rootNode, name="AnimationManager"))

    model = rootNode.addChild('model')
    model.addObject('EulerImplicitSolver')
    model.addObject('SparseLDLSolver', template = "CompressedRowSparseMatrixd")
    model.addObject('GenericConstraintCorrection')

    model.addObject('MeshVTKLoader', name='loader', filename=pathMesh+'full_quadriped_fine.vtk')
    model.addObject('TetrahedronSetTopologyContainer', src='@loader')
    model.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false', showIndicesScale=4e-5,
                    rx=0, printLog=False)
    model.addObject('UniformMass', name='quadrupedMass', totalMass=0.035, printLog=False)
    model.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.05,  youngModulus= 70)

    model.addObject('BoxROI', name='membraneROISubTopo', box=[0, 0, -0.1, 150, -100, 0.1], computeTetrahedra=False,
                    drawBoxes=True)
    modelSubTopo = model.addChild('modelSubTopo')
    modelSubTopo.addObject('TriangleSetTopologyContainer', position='@membraneROISubTopo.pointsInROI',
                           triangles="@membraneROISubTopo.trianglesInROI", name='container')
    modelSubTopo.addObject('TriangleFEMForceField', template='Vec3d', name='Append_subTopoFEM',
                           method='large', poissonRatio=0.49,  youngModulus=5000)


    ##########################################
    # Constraint                             #
    ##########################################

    #Inverse

    centerCavity = model.addChild('centerCavity')
    centerCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Center-cavity_finer.stl')
    centerCavity.addObject('MeshTopology', src='@loader', name='topo')
    centerCavity.addObject('MechanicalObject', name='centerCavity')
    centerCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',
                         triangles='@topo.triangles', drawPressure=0, drawScale=0.0002, minPressure= 0.0001, maxPressure = 5, maxPressureVariation = 0.3, initPressure = 0.5)
    centerCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    rearLeftCavity = model.addChild('rearLeftCavity')
    rearLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Left-cavity_finer.stl')
    rearLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    rearLeftCavity.addObject('MechanicalObject', name='rearLeftCavity')
    rearLeftCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',triangles='@topo.triangles', drawPressure=0,
                             drawScale=0.0002, minPressure= 0.0001, maxPressure = 3, maxPressureVariation = 0.3, initPressure = 0.5)
    rearLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')

    rearRightCavity = model.addChild('rearRightCavity')
    rearRightCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Right-cavity_finer.stl')
    rearRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    rearRightCavity.addObject('MechanicalObject', name='rearRightCavity')
    rearRightCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',
                              triangles='@topo.triangles', drawPressure=0, drawScale=0.0002, minPressure= 0.0001, maxPressure = 3, maxPressureVariation =  0.3, initPressure = 0.5)
    rearRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    frontLeftCavity = model.addChild('frontLeftCavity')
    frontLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Front-Left-cavity_finer.stl')
    frontLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    frontLeftCavity.addObject('MechanicalObject', name='frontLeftCavity')
    frontLeftCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',
                              triangles='@topo.triangles', drawPressure=0, drawScale=0.0002, minPressure= 0.0001, maxPressure = 3, maxPressureVariation =  0.3, initPressure = 0.5)
    frontLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')

    frontRightCavity = model.addChild('frontRightCavity')
    frontRightCavity.addObject('MeshSTLLoader', name='loader',
                               filename=pathMesh+'quadriped_Front-Right-cavity_finer.stl')
    frontRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    frontRightCavity.addObject('MechanicalObject', name='frontRightCavity')
    frontRightCavity.addObject('SurfacePressureActuator', name="SurfacePressureActuator", template='Vec3d',
    triangles='@topo.triangles', drawPressure=0, drawScale=0.0002, minPressure= 0.0001, maxPressure = 3, maxPressureVariation =  0.3, initPressure = 0.6)
    frontRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    # #Directe
    # centerCavity = model.addChild('centerCavity')
    # centerCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Center-cavity_finer.stl')
    # centerCavity.addObject('MeshTopology', src='@loader', name='topo')
    # centerCavity.addObject('MechanicalObject', name='centerCavity')
    # centerCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
    #                        value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
    #                        valueType="volumeGrowth")
    # centerCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)
    #
    # rearLeftCavity = model.addChild('rearLeftCavity')
    # rearLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Left-cavity_finer.stl')
    # rearLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    # rearLeftCavity.addObject('MechanicalObject', name='rearLeftCavity')
    # rearLeftCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
    #                          valueType="volumeGrowth", value=0.0000, triangles='@topo.triangles', drawPressure=0,
    #                          drawScale=0.0002)
    # rearLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')
    #
    # rearRightCavity = model.addChild('rearRightCavity')
    # rearRightCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Right-cavity_finer.stl')
    # rearRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    # rearRightCavity.addObject('MechanicalObject', name='rearRightCavity')
    # rearRightCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
    #                           value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
    #                           valueType="volumeGrowth")
    # rearRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)
    #
    # frontLeftCavity = model.addChild('frontLeftCavity')
    # frontLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Front-Left-cavity_finer.stl')
    # frontLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    # frontLeftCavity.addObject('MechanicalObject', name='frontLeftCavity')
    # frontLeftCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
    #                           value=0.0000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
    #                           valueType="volumeGrowth")
    # frontLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')
    #
    # frontRightCavity = model.addChild('frontRightCavity')
    # frontRightCavity.addObject('MeshSTLLoader', name='loader',
    #                            filename=pathMesh+'quadriped_Front-Right-cavity_finer.stl')
    # frontRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    # frontRightCavity.addObject('MechanicalObject', name='frontRightCavity')
    # frontRightCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
    #                            value=0.0000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
    #                            valueType="volumeGrowth")
    # frontRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)


    modelCollis = model.addChild('modelCollis')
    modelCollis.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_collision.stl',
                               rotation=[0, 0, 0], translation=[0, 0, 0])
    modelCollis.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    modelCollis.addObject('MechanicalObject', name='collisMO', template='Vec3d')
    modelCollis.addObject('TriangleCollisionModel', group=0)
    modelCollis.addObject('LineCollisionModel', group=0)
    modelCollis.addObject('PointCollisionModel', group=0)
    modelCollis.addObject('BarycentricMapping')

    modelVisu = model.addChild('visu')
    modelVisu.addObject('MeshSTLLoader', name='loader', filename=pathMesh+"quadriped_collision.stl")
    modelVisu.addObject('OglModel', src='@loader', template='Vec3d', color=[0.7, 0.7, 0.7, 0.6])
    modelVisu.addObject('BarycentricMapping')

    planeNode = rootNode.addChild('Plane')

    planeNode.addObject('MeshOBJLoader', name='loader', filename="mesh/floorFlat.obj", triangulate="true")
    planeNode.addObject('MeshTopology', src="@loader")
    planeNode.addObject('MechanicalObject', src="@loader", rotation=[90, 0, 0], translation=[250, 35, -1], scale=15)

    planeNode.addObject('TriangleCollisionModel', simulated=0, moving=0, group=1)
    planeNode.addObject('LineCollisionModel', simulated=0, moving=0, group=1)
    planeNode.addObject('PointCollisionModel', simulated=0, moving=0, group=1)

    source = config["source"]
    target = config["target"]
    rootNode.addObject("LightManager")
    spotLoc = [0, 0, 1000]
    rootNode.addObject("SpotLight", position=spotLoc, direction=[0, 0.0, -1.0])
    rootNode.addObject("InteractiveCamera", name="camera", position=source, lookAt=target, zFar=500)


    # Goals
    with open("./TargetPoint.txt", 'r') as outfile:
        targetPointsPos = json.load(outfile)
    init_pos = np.array(targetPointsPos[0])
    translation = [55.26272867193276, -53.71608077293722, 17.499841404638524]
    init_pos[:,0]+= translation[0]
    init_pos[:,1]+= translation[1]
    init_pos[:,2]+= translation[2]
    init_pos = init_pos.tolist()

    goal = rootNode.addChild('Goals')
    GoalMO = goal.addObject('MechanicalObject', name='GoalMO', position=init_pos, showObject = True, showObjectScale=10, showColor = [0, 0, 1, 1])
    rootNode.addObject(MoveGoal(name="MoveGoal", root = rootNode, positions = targetPointsPos))


    # Effectors
    effector = modelCollis.addChild('Effectors')
    effector.addObject('MechanicalObject', position = init_pos, name="EffectorMO", showObject=True, showObjectScale=10, showColor = [1, 0, 0, 1])
    effector.addObject('PositionEffector', template='Vec3', effectorGoal= init_pos, indices = [i for i in range(len(init_pos))])
    effector.addObject('BarycentricMapping')

    # print(">> Add runSofa visualisation")
    # rootNode.addObject(ApplyAction(name="ApplyAction", root = rootNode, actions = [[-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
    #                             [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
    #                             [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]*100, scale = 30) )


    return rootNode

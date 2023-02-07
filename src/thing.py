"""
Class for defining & initializing object meshes in Pybullet simulations.
"""


import pybullet as p
import os
import pybullet_data


class Thing:     

# # neccesary parameters of the mug, the global scaling is used to shrink the size of the mug
# mug:
#   object_name: Mug
#   object_position: [0.7, 0, 0.0]
#   global_scaling : 0.5


    def __init__(self, name, position, globalScaling):

        self.name = name
        self.initPos = position
        self.objectScale = globalScaling    # also defined in the arg85.yaml file, for shrinking the size of the mug
        self.initOrientation = p.getQuaternionFromEuler([0,0,0])

        """
        for better organization, should move the urdf path to the arg85 yaml file as well...
        """
        self.urdf_path = "./src/urdf/objects/mug/mug.urdf"

        print(f"Created a {name}")

        if name=="Mug":
            id = p.loadURDF(self.urdf_path, self.initPos, self.initOrientation, useFixedBase=False, globalScaling = self.objectScale)
            print(f'Load {name}')
            print(f"Set ID to {id}")
            self.setID(id)


        # elif name=="Bottle":
        #     self.objShapeID = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=address, 
        #                             rgbaColor=[1, 0, 0, 1], specularColor=[0.4, .4, 0], 
        #                             visualFramePosition=shift, meshScale=meshScale)
        #     self.objCollisionID = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=address, 
        #                                 collisionFramePosition=shift, meshScale=meshScale)
        #     if onObject is not None:
        #         if onObject.name == "table"  :
        #             self.initPos[2] += onObject.getHeight()

        #     self.ID = p.createMultiBody(baseMass=0.2,
        #                         baseInertialFramePosition=inertial,
        #                         baseCollisionShapeIndex=self.objCollisionID,
        #                         baseVisualShapeIndex=self.objShapeID,
        #                         basePosition=self.initPos,
        #                         baseOrientation=ObjIniOrientation,
        #                         useMaximalCoordinates=True)
            
        #     print(f"Set ID to {self.ID}") 

    def setInitPos(self,pos):
        self.initPos=pos

    def getInitPos(self):
        return self.initPos

    def setID(self,ID):
        self.ID=ID

    def getPos(self):
        bpao=p.getBasePositionAndOrientation(self.ID)
        return bpao[0]

    def setHeight(self,height):
        self.height=height

    def getHeight(self):
        return self.height

    def setWidth(self,width):
        self.width=width

    def getWidth(self):
        return self.width

    def setInitOrientation(self, quat):
        self.initOrientation=quat
    
    def resetObject(self):
        pos = (0, 0, 0)
        orn = p.getQuaternionFromEuler([0,0,0])
        p.resetBasePositionAndOrientation(self.ID, pos, orn)

    def getInitOrientation(self):
        return self.initOrientation 
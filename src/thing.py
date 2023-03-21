"""
Class for defining & initializing object meshes in Pybullet simulations.
"""

import pybullet as p
import os
import pybullet_data


class Thing:
    def __init__(self, name, position, globalScaling):

        self.name = name
        self.initPos = position
        self.objectScale = globalScaling    # also defined in the arg85.yaml file, for shrinking the size of the mug
        self.initOrientation = p.getQuaternionFromEuler([0,0,0])

        """
        for better organization, should move the urdf path to the arg85 yaml file as well...
        """

        if name == "block" or name == "block1":
            self.urdf_path = "./src/urdf/objects/block1/block1.urdf"
        elif name == "block2":
            self.urdf_path = "./src/urdf/objects/block2/block2.urdf"
        elif name == "block3":
            self.urdf_path = "./src/urdf/objects/block3/block3.urdf"

        id = p.loadURDF(self.urdf_path, self.initPos, self.initOrientation, useFixedBase=False, globalScaling=self.objectScale)
        self.setID(id)
        self.resetVelocity()

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
    
    def resetVelocity(self):
        p.resetBaseVelocity(self.ID, [0, 0, 0], [0, 0, 0])
    
    def resetObject(self):
        pos = (0, 0, 0)
        orn = p.getQuaternionFromEuler([0,0,0])
        p.resetBasePositionAndOrientation(self.ID, pos, orn)
        self.resetVelocity()

    def getInitOrientation(self):
        return self.initOrientation 
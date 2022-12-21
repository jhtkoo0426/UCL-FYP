"""
Test script for running a Pybullet simulation
"""

import pybullet as pb
import time
import pybullet_data


physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -10)

planeId = pb.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = pb.getQuaternionFromEuler([0,0,0])
boxId = pb.loadURDF("r2d2.urdf", startPos, startOrientation)

# Center of mass frame
for i in range(10000):
    pb.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
pb.disconnect()
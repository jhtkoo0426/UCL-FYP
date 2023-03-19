"""
Pybullet simulation for Robust Robotic Grasping Utilising Touch Sensing.
"""

import time
import os
import numpy as np
import pybullet as p
import pybullet_data
import tacto

from utilities import Camera
from tqdm import tqdm

from pointCloud import getPointCloud
from thing import Thing



SIMULATION_STEP_DELAY = 1 / 20000.


class Simulator:
    def __init__(self, robot, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        self.p_bar = tqdm(ncols=0, disable=False) if self.vis else None
        self.camera = camera
    
    
import os

import numpy as np
import pybullet as p

from simulation.simulation import Simulator
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
import tacto
import math



def run_simulation():
    # Load YCB models
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )

    # Initialize camera
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    # camera = None

    # Initialize robot
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))

    # Initialize simulation environment
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)
    env.reset()

    env.container.resetObject()

    while True:
        env.step(env.read_debug_parameter(), 'end')
        env.digit_step()


if __name__ == '__main__':
    run_simulation()

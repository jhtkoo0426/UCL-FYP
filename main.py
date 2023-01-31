import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
import time
import math

import cv2
import tacto
import pybulletX as px

from agent import RandomAgent



def user_control_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))

   
    
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)

    env.reset()
    # env.SIMULATION_STEP_DELAY = 0

    digits = tacto.Sensor(**robot.tacto_info, background = robot.bg)

    p.resetDebugVisualizerCamera(**robot.camera_info)

    digits.add_camera(robot.id, robot.link_ID)

    cam1_pos, cam1_orient = env.digits.cameras["cam1"].get_pose()

    print("camera position of digit is", cam1_pos, cam1_orient)

    # Initialize agent to collect sensory data
    

    # mug_start_pos = [0, -0.2, 0.5]
    # mug_start_orientation_euler = [0, 0, 0]
    # mug_start_orientation = p.getQuaternionFromEuler(mug_start_orientation_euler)

    # mug = px.Body("urdf/objects/mug/mug.urdf",use_fixed_base=True, global_scaling = 0.5)
    # # print("mug px file is",mug)
    # mug.set_base_pose(mug_start_pos, mug_start_orientation)

    # digits.add_body(env.container) 
    # digits.add_object(env.container.urdf_path, env.container.id, env.container.objectScale)

    while True:
        obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
        env.digit_step()
        # color, depth = digits.render()
        # # print("depth is", depth)
        # digits.updateGUI(color, depth)
        # print(obs, reward, done, info)


if __name__ == '__main__':
    user_control_demo()

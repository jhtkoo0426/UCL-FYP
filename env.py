import time
import math
import random
import os

import numpy as np
import pybullet as p
import pybullet_data
import tacto

from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm
from pointCloud import getPointCloud
from thing import Thing
from robot import RobotBase


class FailToReachTargetError(RuntimeError):
    pass


class ClutteredPushGrasp:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        # call the function to read the arg85 yaml file
        self.robot.load_digit_parm()
        # add the digit sensor to the arg85 hand

        self.digits = tacto.Sensor(**self.robot.tacto_info, background = self.robot.bg)
        p.resetDebugVisualizerCamera(**self.robot.camera_info)
        self.digits.add_camera(self.robot.id, self.robot.link_ID)
        

        
        # added point cloud button and initial button values
        self.initButtonVals()
        self.pointCloudButton = p.addUserDebugParameter("Get point cloud", 1, 0, 1)

        # added digit button to save the frame to a matrix...?
        self.DigitTempSaveButton = p.addUserDebugParameter("Save digit frame temp", 1, 0, 1)

        self.DigitSaveButton = p.addUserDebugParameter("Save digit frame local", 1, 0, 1)

        # LOAD the mug (or other objects later into the envrionment)

        self.container = Thing(robot.object_info["object_name"], robot.object_info["object_position"], robot.object_info["global_scaling"])

        # Load the mug to the tacto digit sensor

        self.digits.add_object(self.container.urdf_path, self.container.id, self.container.objectScale)

        # added the get point cloud class... maybe to make it inherit the properties of the class?
        # self.PC = getPointCloud(target = robot.object_info["object_position"])

        """
        original function to load a box into the physical envrionment
        """
        self.boxID = p.loadURDF("./urdf/skew-box-button.urdf",
                                [0.0, 0.0, 0.0],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True,
                                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

        # For calculating the reward
        self.box_opened = False
        self.btn_pressed = False
        self.box_closed = False




    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length


    def initButtonVals(self):
        # set the button value for point cloud
        self.pointCloudButtonVal = 2.0
        self.DigitTempSaveButtonVal = 2.0
        self.DigitSaveButtonVal = 2.0

    def readPointCloudButton(self):
        if p.readUserDebugParameter(self.pointCloudButton) >= self.pointCloudButtonVal:
            print("work!")
            # call the function to get point cloud

            # pcl = self.PC.getPointCloud(self.container.getPos(), self.getPandO())
            
            pcl = getPointCloud(target = self.robot.object_info["object_position"])
            self.robot.setPointcloud(pcl)
  
            self.pointCloudButtonVal = p.readUserDebugParameter(self.pointCloudButton) + 1.0

    def readDigitTempSaveButton(self):
        if p.readUserDebugParameter(self.DigitTempSaveButton) >= self.DigitTempSaveButtonVal:
            print("appending the frame to the list")
            self.robot.digit_depth_img.append(self.depth)
            self.robot.digit_RGB_img.append(self.color)
        self.DigitTempSaveButtonVal = p.readUserDebugParameter(self.DigitTempSaveButton) + 1.0

    def readDigitSaveButton(self):
        if p.readUserDebugParameter(self.DigitSaveButton) >= self.DigitSaveButtonVal:
            print("saving the frame to local folder")
            curr_dir = os.getcwd()
            Digit_folder = "Digit_data/mug"
            if not os.path.exists(Digit_folder):
                os.mkdir(os.path.join(curr_dir,Digit_folder))
                print("creating folder",Digit_folder)


            # specified the local folder and file name 
            # RGB
            Digit_RGB_data = "RGB_camera_frame.npy"
            digit_data_path_RGB = os.path.join(curr_dir,Digit_folder,Digit_RGB_data)
            np.save(digit_data_path_RGB,np.asarray(self.robot.digit_RGB_img))

            # depth
            Digit_depth_data = "depth_camera_frame.npy"
            digit_data_path_depth = os.path.join(curr_dir,Digit_folder,Digit_depth_data)
            # save the matrix as a local npy file
            np.save(digit_data_path_depth,np.asarray(self.robot.digit_depth_img))

        self.DigitSaveButtonVal = p.readUserDebugParameter(self.DigitSaveButton) + 1.0

    # def getPandO(self):
    #     linkPos = 13    # This corresponds to the link id of the digit sensor, "joint_digit_1.0_tip', mounted to the left finger
    #     com_p, com_o, _, _, _, _ = p.getLinkState(self.robot, linkPos, computeForwardKinematics=True)
        
    #     com_o_Eul=p.getEulerFromQuaternion(com_o)
    #     #print(com_o_Eul)
    #     normtemp=np.linalg.norm(com_o_Eul)
        
    #     #quickfix 
    #     com_p1= com_p[0]-(0.2/normtemp)*com_o_Eul[0]
    #     com_p2= com_p[1]-(0.2/normtemp)*com_o_Eul[1]
    #     com_p3= com_p[2]-(0.2/normtemp)*com_o_Eul[2]
    #     com_p=(com_p1,com_p2,com_p3)
    #     return com_p

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])

        # in the step simulation, read the point cloud
        self.readPointCloudButton()
        # in the step simulation, update the renderer of tacto sensor


        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        reward = self.update_reward()
        done = True if reward == 1 else False
        info = dict(box_opened=self.box_opened, btn_pressed=self.btn_pressed, box_closed=self.box_closed)
        return self.get_observation(), reward, done, info

    def digit_step(self):
        self.color, self.depth = self.digits.render()
        self.digits.updateGUI(self.color, self.depth)

        # check whether the frame should be saved to a list
        self.readDigitTempSaveButton()

        # check whether the list of renderer frame should be saved locally
        self.readDigitSaveButton()
        

    def update_reward(self):
        reward = 0
        if not self.box_opened:
            if p.getJointState(self.boxID, 1)[0] > 1.9:
                self.box_opened = True
                print('Box opened!')
        elif not self.btn_pressed:
            if p.getJointState(self.boxID, 0)[0] < - 0.02:
                self.btn_pressed = True
                print('Btn pressed!')
        else:
            if p.getJointState(self.boxID, 1)[0] < 0.1:
                print('Box closed!')
                self.box_closed = True
                reward = 1
        return reward

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def reset_box(self):
        p.setJointMotorControl2(self.boxID, 0, p.POSITION_CONTROL, force=1)
        p.setJointMotorControl2(self.boxID, 1, p.VELOCITY_CONTROL, force=0)

    def reset(self):
        self.robot.reset()
        self.reset_box()
        return self.get_observation()

    def close(self):
        p.disconnect(self.physicsClient)

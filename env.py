import time
import os

import numpy as np
import pybullet as p
import pybullet_data
import tacto

from utilities import Models, Camera
from tqdm import tqdm
from pointCloud import getPointCloud
from thing import Thing
from robot import RobotBase




class ClutteredPushGrasp:
    # Global constants
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

        # Read the parameters.yaml file to load robot parameters
        self.robot.load_digit_parm()

        # Mount DIGIT tactile sensors to hand
        self.digits = tacto.Sensor(**self.robot.tacto_info, background = self.robot.bg)
        p.resetDebugVisualizerCamera(**self.robot.camera_info)
        self.digits.add_camera(self.robot.id, self.robot.link_ID)

        
        # Point cloud button and initial button values
        self.initButtonVals()
        self.pointCloudButton = p.addUserDebugParameter("Get point cloud", 1, 0, 1)

        # DIGIT button to save the tactile readings as numpy array
        self.DigitTempSaveButton = p.addUserDebugParameter("Save digit frame temp", 1, 0, 1)
        self.DigitSaveButton = p.addUserDebugParameter("Save digit frame local", 1, 0, 1)

        # Button to get current joint coordinates and 6d pose of end effector
        self.jointObsButton = p.addUserDebugParameter("Get joint coordinates", 1, 0, 1)
        self.fetch6dButton = p.addUserDebugParameter("Get 6d pose", 1, 0 ,1)

        # Button for data collection
        self.dataCollectionButton = p.addUserDebugParameter("Collect sensory data...", 1, 0, 1)

        # Load the target object defined in the parameters.yaml file (or other objects later into the envrionment)
        self.container = Thing(self.robot.object_info["object_name"], self.robot.object_info["object_position"], self.robot.object_info["global_scaling"])

        # Load the mug to the tacto digit sensor
        # self.digits.add_object(self.container.urdf_path, self.container.id, self.container.objectScale)

        # added the get point cloud class... maybe to make it inherit the properties of the class?
        # self.PC = getPointCloud(target = robot.object_info["object_position"])

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
        self.jointObsButtonVal = 2.0
        self.fetch6dButtonVal = 2.0
        self.dataCollectionButtonVal = 2.0

    def readPointCloudButton(self):
        if p.readUserDebugParameter(self.pointCloudButton) >= self.pointCloudButtonVal:
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

    def readJointObsButton(self, sixd_pose):
        if p.readUserDebugParameter(self.jointObsButton) >= self.DigitSaveButtonVal:
            print(f"6D Pose: {sixd_pose}")
        self.jointObsButtonVal = p.readUserDebugParameter(self.jointObsButton) + 1.0
    
    def readFetch6dButton(self, action):
        if p.readUserDebugParameter(self.fetch6dButton) >= self.fetch6dButtonVal:
            print(f"Action: {action}")
        self.fetch6dButtonVal = p.readUserDebugParameter(self.fetch6dButton) + 1.0
    
    # Run data collection code via button onclick
    def readDataCollectionButton(self):
        print("Data Collection Button read...")
        position = np.asarray([0.0, 0.0, 0.1631578952074051, 0.0, 1.570796251296997, 1.5707963705062866])

        random_poses_count = 200

        # Nx6 numpy array (6d poses)
        random_poses = np.zeros(shape=(random_poses_count, 6))

        # [Flattened actile data 160 * 240 + Grasp result (1)] + N trials
        generated_training_dataset = np.zeros(shape=(230401, random_poses_count))

        #  Add a bit of height to poses. Prevents the robot from colliding with the object when moving to it
        z_padding = abs(0.5)

        # Set robot movement speed
        velocity_scale = 0.3

        if p.readUserDebugParameter(self.dataCollectionButton) >= self.dataCollectionButtonVal:
            # Generate random poses using Gaussian noise
            for i in range(random_poses_count):
                # Add noise to rpy only
                gaussian_1d_noise = np.random.normal(0, 0.05, 3)
                noisy_rpy = np.add(position[-3:], gaussian_1d_noise)
                noisy_position = np.concatenate([position[:3], noisy_rpy])

                # Increase z by z_padding to prevent robot-object collision
                noisy_position[2] = noisy_position[2] + z_padding
                random_poses[i] = noisy_position
            print(f"Generated random poses of shape {random_poses.shape}")

            # Execute generated grasps
            for i in range(len(random_poses)):
                print(f"Random pose {str(i+1)} | {random_poses[i]}")
                sixd_pose = tuple(random_poses[i])

                # Move arm to pose
                self.robot.move_ee_data_col(sixd_pose, 'end', velocity_scale)
                for _ in range(500):  # Wait for a few steps
                    self.step_simulation()

                # Open gripper
                self.robot.open_gripper()

                # Lower the arm by z=2
                lower_sixd_pose = random_poses[i].copy()
                lower_sixd_pose[2] = lower_sixd_pose[2] - z_padding
                self.robot.move_ee_data_col(lower_sixd_pose, 'end', velocity_scale)
                for _ in range(500):  # Wait for a few steps
                    self.step_simulation()

                # Perform grasp (close gripper)
                self.robot.close_gripper()

                # Record tactile data
                print(np.asarray(self.depth).shape, np.asarray(self.color).shape)
                tactile_data = np.array(self.color).flatten()

                # Lift object for 5s
                upper_sixd_pose = random_poses[i].copy()
                upper_sixd_pose[2] = upper_sixd_pose[2] + z_padding
                self.robot.move_ee_data_col(upper_sixd_pose, 'end', velocity_scale)

                # Record object position to determine if it moved after a while
                grasped_object_position = self.container.getPos()
                for _ in range(1500):  # Wait for a few steps
                    self.step_simulation()

                # Record success/failure
                final_object_position = self.container.getPos()

                grasp_outcome = None
                if grasped_object_position == final_object_position and final_object_position != self.container.getInitPos:
                    print('SUCCESSFUL GRASP')
                    grasp_outcome = 1
                else:
                    print('UNSUCCESSFUL GRASP')
                    grasp_outcome = 0
                print(len(tactile_data))
                generated_training_dataset[i] = np.concatenate([tactile_data, grasp_outcome])

                # Reset simulation (or reset object position)
                self.reset_simulation()

                for _ in range(100):
                    self.step_simulation()

        self.dataCollectionButtonVal = p.readUserDebugParameter(self.dataCollectionButton) + 1.0

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

        # in the step simulation, get the joint orientation
        sixd = self.robot.get_joint_obs()

        # in the step simulation, read the point cloud
        self.readPointCloudButton()
        self.readJointObsButton(sixd)
        self.readFetch6dButton(action[:-1])
        self.readDataCollectionButton()
        # in the step simulation, update the renderer of tacto sensor

        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        reward, done, info = None, None, None
        return self.get_observation(), reward, done, info

    def digit_step(self):
        self.color, self.depth = self.digits.render()
        self.digits.updateGUI(self.color, self.depth)

        # check whether the frame should be saved to a list
        self.readDigitTempSaveButton()

        # check whether the list of renderer frame should be saved locally
        self.readDigitSaveButton()
        

    def update_reward(self):
        pass

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def reset(self):
        self.robot.reset()
        # self.reset_box()
        return self.get_observation()
    
    # Reset the whole simulation
    def reset_simulation(self):
        self.robot.reset()
        self.container.resetObject()

    def close(self):
        p.disconnect(self.physicsClient)

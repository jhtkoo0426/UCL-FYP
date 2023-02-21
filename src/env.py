import time
import os

import numpy as np
import pybullet as p
import pybullet_data
import tacto
import cv2

from utilities import Models, Camera
from tqdm import tqdm
from pointCloud import getPointCloud
from thing import Thing
from robot import RobotBase


import matplotlib.pyplot as plt



class ClutteredPushGrasp:
    # Global constants
    SIMULATION_STEP_DELAY = 1 / 20000.


    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis

        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # Define Pybullet simulation environment
        # p.GUI for a simulation w/ GUI, p.DIRECT for a headless simulation
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # Reset debug parameters
        self.resetUserDebugParameters()

        # Read the parameters.yaml file to load robot parameters
        self.robot.load_digit_parm()

        # Mount DIGIT tactile sensors to hand
        self.digits = tacto.Sensor(**self.robot.tacto_info, background = self.robot.bg)
        p.resetDebugVisualizerCamera(**self.robot.camera_info)
        self.digits.add_camera(self.robot.id, self.robot.link_ID)

        # Load the target object defined in the parameters.yaml file (or other objects later into the envrionment)
        self.container = Thing(self.robot.object_info["object_name"], self.robot.object_info["object_position"], self.robot.object_info["global_scaling"])

        # Load the mug to the tacto digit sensor
        self.digits.add_object(self.container.urdf_path, self.container.ID, self.container.objectScale)

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
        self.openGripperButtonVal = 2.0
        self.closeGripperButtonVal = 2.0
        self.resetSimulationButtonVal = 2.0

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
            Digit_folder = "./src/Digit_data/mug"
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
    
    def readOpenGripperButton(self):
        if p.readUserDebugParameter(self.openGripperButton) >= self.openGripperButtonVal:
            self.robot.open_gripper()
        self.openGripperButtonVal = p.readUserDebugParameter(self.openGripperButton) + 1.0
    
    def readCloseGripperButton(self):
        if p.readUserDebugParameter(self.closeGripperButton) >= self.closeGripperButtonVal:
            self.robot.close_gripper()
        self.closeGripperButtonVal = p.readUserDebugParameter(self.closeGripperButton) + 1.0

    def readJointObsButton(self, angles):
        if p.readUserDebugParameter(self.jointObsButton) >= self.DigitSaveButtonVal:
            print(f"Joint angles (length={len(angles)}): {angles}")
        self.jointObsButtonVal = p.readUserDebugParameter(self.jointObsButton) + 1.0
    
    def readFetch6dButton(self, action):
        if p.readUserDebugParameter(self.fetch6dButton) >= self.fetch6dButtonVal:
            print(f"End effector 6D pose: {action}")
        self.fetch6dButtonVal = p.readUserDebugParameter(self.fetch6dButton) + 1.0
    
    def readResetSimulationButton(self):
        if p.readUserDebugParameter(self.resetSimulationButton) >= self.resetSimulationButtonVal:
            self.reset_simulation()
        self.resetSimulationButtonVal = p.readUserDebugParameter(self.resetSimulationButton) + 1.0
    
    # Auxiliary function for readDataCollectionButton to generate random poses
    def generateGaussianNoisePoses(self, target_poses_count, base_6d_poses, z_padding):
        """
        Generates random end effector poses following a Gaussian distribution.

        @param target_poses_count: (int) indicating the number of poses to generate
        @param base_6d_poses: (1D numpy array) representing the base end effector pose to apply gaussian noise to
        @param z_padding: (float) slight padding to prevent the robot from colliding with the object when moving
                          to it via inverse kinematics
        
        @returns random_poses: filled array of generated end effector poses
        """

        random_poses = []

        for _ in range(target_poses_count):
            # Add noise to 6d pose (x,y,z,r,p,y)
            # Block noise
            # x_noise = np.random.normal(0, 0.005, 1).item()
            # y_noise = np.random.normal(0, 0.005, 1).item()
            # z_noise = np.random.normal(0, 0.01, 1).item() + z_padding
            # or_noise = 0
            # op_noise = np.random.normal(0, 0.1, 1).item()
            # oy_noise = np.random.normal(0, 0.1, 1).item()

            # Bottle noise
            x_noise = np.random.normal(0, 0.01, 1).item()
            y_noise = np.random.normal(0, 0.01, 1).item()
            z_noise = np.random.normal(0, 0.1, 1).item() + z_padding
            or_noise = 0
            op_noise = np.random.normal(0, 0.07, 1).item()
            oy_noise = np.random.normal(0, 0.07, 1).item()
            
            # Apply the Gaussian noise for each base pose
            for base_6d_pose in base_6d_poses:
                noisy_pose = np.array([x_noise, y_noise, z_noise, or_noise, op_noise, oy_noise])
                noisy_pose = base_6d_pose + noisy_pose
                random_poses.append(noisy_pose)
            
            if len(random_poses) == target_poses_count:
                break
        
        return np.array(random_poses)
    
    # Auxiliary function as sanity check for collecting tactile readings
    def tactileSanityCheck(self, depth, color):
        """
        Checks if the collected tactile data actually represent any grasps. Discards the data if the sensor
        doesn't pick up any readings.

        @param depth: (np.array) a pair of depth tactile readings from the tactile sensor; Shape: (2, 160, 120)
        @param color: (np.array) a pair of color tactile readings from the tactile sensor; Shape: (2, 160 , 120, 3)
        """

        print(len(np.unique(depth[0], return_counts=True)[0]) == 1, len(np.unique(depth[1], return_counts=True)[0]) == 1, len(np.unique(color[0], return_counts=True)[0]) == 1, len(np.unique(color[1], return_counts=True)[0]) == 1)

        if len(np.unique(depth[0], return_counts=True)[0]) == 1 or len(np.unique(depth[1], return_counts=True)[0]) == 1 or len(np.unique(color[0], return_counts=True)[0]) == 1 or len(np.unique(color[1], return_counts=True)[0]) == 1:
            print("Found invalid depth and color readings. Skipping this set of readings...")
            return None, None
        return depth, color
    
    # Auxiliary function for step_simulation with predetermined step size
    def fixed_step_sim(self, step_size):
        for _ in range(step_size):
            self.step_simulation()
    
    # Auxiliary function to execute a grasp given an end effector pose
    def execute_pose(self, grasp_pose, velocity_scale, z_padding):
        # 1. Move arm to pose and prepare gripper
        self.robot.move_ee_data_col(grasp_pose, 'end', velocity_scale)
        self.robot.open_gripper()
        self.fixed_step_sim(500)

        # 2. Lower the arm by z=2
        lower_sixd_pose = grasp_pose.copy()
        lower_sixd_pose[2] = lower_sixd_pose[2] - z_padding
        self.robot.move_ee_data_col(lower_sixd_pose, 'end', velocity_scale)
        self.fixed_step_sim(500)

        # 3. Close gripper to perform grasp
        self.robot.close_gripper()
        self.fixed_step_sim(500)
        
        # 4. Update the DIGIT camera to collect color and depth of DIGIT sensor
        self.digit_step()
        self.fixed_step_sim(500)

        # 5. Record tactile data
        color = np.asarray(self.color)      # (2, 160, 120, 3)
        depth = np.asarray(self.depth)      # (2, 160, 120)

        # fig = plt.figure(figsize=(5,2))
        # fig.add_subplot(1,4,1)
        # plt.imshow(depth[0], cmap='gray')
        # fig.add_subplot(1,4,2)
        # plt.imshow(color[0])
        # fig.add_subplot(1,4,3)
        # plt.imshow(depth[1], cmap='gray')
        # fig.add_subplot(1,4,4)
        # plt.imshow(color[1])
        # plt.savefig('test.png')

        # color = np.concatenate(self.color, axis=1)                                                  # 1. Concatenate colors horizontally (axis=1)
        # depth = np.concatenate([self.digits._depth_to_color(d) for d in self.depth], axis=1)        # 2. Convert depth to color
        # color_n_depth = np.concatenate([color, depth], axis=0)                                      # 3. Concatenate the resulting 2 images vertically (axis=0)
        # # tactile_data = np.array(color_n_depth).flatten()                                            # 4. Flatten image to put into dataset
        # tactile_data = np.array(color_n_depth)

        # 6. Lift object for 5s to determine successful vs unsuccessful grasp
        upper_sixd_pose = grasp_pose.copy()
        upper_sixd_pose[2] += z_padding
        self.robot.move_ee_data_col(upper_sixd_pose, 'end', velocity_scale)

        # Record object z position to determine if it moved after a while
        grasped_object_z_pos = self.container.getPos()[2]
        self.fixed_step_sim(1500)

        # Record success/failure
        final_object_z_pos = self.container.getPos()[2]

        # Determine if the grabbed object stays in the same z position AND the z position is not 0 (as defined in container.getInitPos())
        delta_z = final_object_z_pos - grasped_object_z_pos
        grasp_outcome = np.ones(shape=(1,)) if delta_z > z_padding and final_object_z_pos > 0 else np.zeros(shape=(1,))
        return color, depth, grasp_outcome

    # Run data collection code via button onclick
    def readDataCollectionButton(self):
        """
        This function executes a data collection loop by generating N gaussian-distributed end effector poses,
        then collecting the corresponding DIGIT tactile sensor readings on each finger of the gripper (as the 
        tactile data) as well as the end effector poses (as the visual data).

        The collected data is then used for stability classification in which the project aims to find the
        best representation of this data. This serves as the basis for further work on learning a generative
        model.
        """
        
        # Block positions
        # positions = np.array([[0.0, 0.0, 0.1725230525032501, 0.0, 1.570796251296997, 1.5707963705062866],
        #                       [0.0, -0.05187368392944336, 0.1984210538864136, 0.0, 1.3221051692962646, 1.5707963705062866],
        #                       [-0.007073685526847839, 0.028294742107391357, 0.16842105984687805, 0.0, 1.7187368869781494, 1.5707963705062866]])
        # Bottle positions
        positions = np.array([[0.1296842396259308, 0.014147371053695679, 0.13684210181236267, 0.09915781021118164, 1.520420789718628, 0.5952490568161011],
                              [0.10374736785888672, -0.01886315643787384, 0.1315789520740509, 0.0, 1.553473711013794, 0.8928736448287964],
                              [0.16033685207366943, 0.06602105498313904, 0.057894736528396606, 0.0, 1.570796251296997, 0.5952490568161011]])
        random_poses_count = 400

        # Separate data arrays for tactile and visual data
        valid_random_poses = np.empty((0, 6))
        depth_dataset = np.empty((0, 2, 160, 120))       # N trials x [Tactile data (depth) 160x120]
        color_dataset = np.empty((0, 2, 160, 120, 3))    # N trials x [Tactile data (color) 160x120x3]
        grasp_outcomes = np.empty((0))                   # N trials

        # Parameters of robot setup (can be changed)
        Z_PADDING = abs(0.2)        # Prevents the robot from colliding with the object when moving to it
        VELOCITY_SCALE = 0.15       # Scales the robot movement speed

        successes = 0
        fails = 0

        if p.readUserDebugParameter(self.dataCollectionButton) >= self.dataCollectionButtonVal:
            # Generate random poses using Gaussian noise
            random_poses = self.generateGaussianNoisePoses(random_poses_count, positions, Z_PADDING)
            print(f"Generated random poses of shape {random_poses.shape}")

            # Execute generated grasps
            for i in range(len(random_poses)):
                sixd_pose = np.array(random_poses[i])
                print(f"Random pose {str(i+1)}: {sixd_pose}")

                # Execute grasp to get tactile readings and outcome
                color, depth, grasp_outcome = self.execute_pose(sixd_pose, VELOCITY_SCALE, Z_PADDING)

                if grasp_outcome[0] == 1:
                    successes += 1
                    print("SUCCESS")
                else:
                    fails += 1
                    print("FAIL")
                print(f"Successes: {successes} | Fails: {fails}")
                

                # Sanity check to make sure the data is valid
                depth, color = self.tactileSanityCheck(depth, color)

                if depth is None or color is None:
                    print(f"Not saving pose {str(i)} data to dataset.")
                else:
                    # Save to dataset
                    valid_random_poses = np.append(valid_random_poses, [random_poses[i]], axis=0)
                    depth_dataset = np.append(depth_dataset, [depth], axis=0)
                    color_dataset = np.append(color_dataset, [color], axis=0)
                    grasp_outcomes = np.append(grasp_outcomes, grasp_outcome, axis=0)
                    print(f"Saved data from pose {str(i)} to dataset.")

                # 7. Reset robot and arm only
                self.robot.reset()
                self.container.resetObject()
                self.fixed_step_sim(500)
            
            # Save generated data to .npy files
            CURR_DIR = os.getcwd()
            BASELINE_DIR = "./src/baseline_model"

            # Save depth data to npy file
            depth_ds_filename = "depth_ds.npy"
            depth_ds_path = os.path.join(CURR_DIR, BASELINE_DIR, depth_ds_filename)
            np.save(depth_ds_path, depth_dataset)
            print(f"Training data saved to {depth_ds_path}")

            # Save color data to npy file
            color_ds_filename = "color_ds.npy"
            color_ds_path = os.path.join(CURR_DIR, BASELINE_DIR, color_ds_filename)
            np.save(color_ds_path, color_dataset)
            print(f"Training data saved to {color_ds_path}")

            # Save end effector poses to npy file
            poses_ds_filename = "poses_ds.npy"
            poses_ds_path = os.path.join(CURR_DIR, BASELINE_DIR, poses_ds_filename)
            np.save(poses_ds_path, valid_random_poses)
            print(f"End effector poses saved to {poses_ds_path}")

            # Save grasp outcome to npy file
            grasp_ds_filename = "grasp_outcomes.npy"
            grasp_ds_path = os.path.join(CURR_DIR, BASELINE_DIR, grasp_ds_filename)
            np.save(grasp_ds_path, grasp_outcomes)
            print(f"Grasp outcomes saved to {grasp_ds_path}")


        self.dataCollectionButtonVal = p.readUserDebugParameter(self.dataCollectionButton) + 1.0

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

        self.readResetSimulationButton()
        self.readJointObsButton(sixd)
        self.readFetch6dButton(action[:-1])
        self.readDataCollectionButton()
        self.readOpenGripperButton()
        self.readCloseGripperButton()
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
        return self.get_observation()
    
    def resetUserDebugParameters(self):
        # Re-initialize sliders
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("Gripper opening length", 0, 0.04, 0.04)

        # Re-initialize buttons
        # Simulation buttons
        self.resetSimulationButton = p.addUserDebugParameter("Reset simulation", 1, 0, 1)
        
        # Point cloud button and initial button values
        self.initButtonVals()
        self.pointCloudButton = p.addUserDebugParameter("Get point cloud", 1, 0, 1)

        # DIGIT button to save the tactile readings as numpy array
        self.DigitTempSaveButton = p.addUserDebugParameter("Save digit frame temp", 1, 0, 1)
        self.DigitSaveButton = p.addUserDebugParameter("Save digit frame local", 1, 0, 1)

        # Button to get current joint coordinates and 6d pose of end effector
        self.jointObsButton = p.addUserDebugParameter("Get joint coordinates", 1, 0, 1)
        self.fetch6dButton = p.addUserDebugParameter("Get end effector pose", 1, 0 ,1)

        # Button for data collection
        self.dataCollectionButton = p.addUserDebugParameter("Collect sensory data", 1, 0, 1)

        # Gripper control
        self.openGripperButton = p.addUserDebugParameter("Open gripper", 1, 0, 1)
        self.closeGripperButton = p.addUserDebugParameter("Close gripper", 1, 0, 1)

    # Reset the whole simulation
    def reset_simulation(self):
        # Remove sliders and buttons
        p.removeAllUserParameters()

        # Re-initialize sliders and buttons
        self.resetUserDebugParameters()

        self.robot.reset()
        self.container.resetObject()

    def close(self):
        p.disconnect(self.physicsClient)

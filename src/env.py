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

        # Import the robot
        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # Prepare parameters
        self.resetUserDebugParameters()
        self.robot.load_digit_parm()        # Read the parameters.yaml file to load robot parameters

        # Mount DIGIT tactile sensors and cameras
        self.digits = tacto.Sensor(**self.robot.tacto_info, background = self.robot.bg)
        p.resetDebugVisualizerCamera(**self.robot.camera_info)
        self.digits.add_camera(self.robot.id, self.robot.link_ID)

        # Load the target object defined in the parameters.yaml file (or other objects later into the envrionment)
        self.object_name        = self.robot.object_info["object_name"]
        self.object_position    = self.robot.object_info["object_position"]
        self.global_scaling     = self.robot.object_info["global_scaling"]
        self.container = Thing(self.object_name, self.object_position, self.global_scaling)

        # Load the specified object to the DIGIT tactile sensors
        self.digits.add_object(self.container.urdf_path, self.container.ID, self.container.objectScale)

    # Hooks to pybullet.stepSimulation()
    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    # Auxiliary function for step_simulation with predetermined step size
    def fixed_step_sim(self, step_size):
        for _ in range(step_size):
            self.step_simulation()
    
    # Read the values of the sliders in the simulation GUI
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

    # Initialize simulation GUI button values
    def initButtonVals(self):
        self.pointCloudButtonVal = 2.0
        self.jointObsButtonVal = 2.0
        self.fetch6dButtonVal = 2.0
        self.dataCollectionButtonVal = 2.0
        self.generativeModelButtonVal = 2.0
        self.openGripperButtonVal = 2.0
        self.closeGripperButtonVal = 2.0
        self.resetSimulationButtonVal = 2.0

    # READ USER DEBUG PARAMETERS (BUTTONS IN GUI) AND EXECUTE CORRRESPONDING ACTIONS
    def readPointCloudButton(self):
        if p.readUserDebugParameter(self.pointCloudButton) >= self.pointCloudButtonVal:
            pcl = getPointCloud(target = self.robot.object_info["object_position"])
            self.robot.setPointcloud(pcl)
        self.pointCloudButtonVal = p.readUserDebugParameter(self.pointCloudButton) + 1.0
    
    def readOpenGripperButton(self):
        if p.readUserDebugParameter(self.openGripperButton) >= self.openGripperButtonVal:
            self.robot.open_gripper()
        self.openGripperButtonVal = p.readUserDebugParameter(self.openGripperButton) + 1.0
    
    def readCloseGripperButton(self):
        if p.readUserDebugParameter(self.closeGripperButton) >= self.closeGripperButtonVal:
            self.robot.close_gripper()
        self.closeGripperButtonVal = p.readUserDebugParameter(self.closeGripperButton) + 1.0

    def readJointObsButton(self, angles):
        if p.readUserDebugParameter(self.jointObsButton) >= self.jointObsButtonVal:
            print(f"Joint angles (length={len(angles)}): {angles}")
        self.jointObsButtonVal = p.readUserDebugParameter(self.jointObsButton) + 1.0
    
    def readFetch6dButton(self, action):
        if p.readUserDebugParameter(self.fetch6dButton) >= self.fetch6dButtonVal:
            print(f"End effector 6D pose: {action}")
        self.fetch6dButtonVal = p.readUserDebugParameter(self.fetch6dButton) + 1.0
    
    def readGenerativeModelButton(self):
        if p.readUserDebugParameter(self.generativeModelButton) >= self.generativeModelButtonVal:
            print("Btn works")
        self.generativeModelButtonVal = p.readUserDebugParameter(self.generativeModelButton) + 1.0
    
    def readResetSimulationButton(self):
        if p.readUserDebugParameter(self.resetSimulationButton) >= self.resetSimulationButtonVal:
            self.reset_simulation()
        self.resetSimulationButtonVal = p.readUserDebugParameter(self.resetSimulationButton) + 1.0
    
    def readDataCollectionButton(self):
        if p.readUserDebugParameter(self.dataCollectionButton) >= self.dataCollectionButtonVal:
            self.collect_data()
        self.dataCollectionButtonVal = p.readUserDebugParameter(self.dataCollectionButton) + 1.0

    def readGenerativeModelButton(self):
        if p.readUserDebugParameter(self.generativeModelButton) >= self.generativeModelButtonVal:
            self.learning_framework()
        self.generativeModelButtonVal = p.readUserDebugParameter(self.generativeModelButton) + 1.0

    
    # HELPER FUNCTIONS FOR DATA COLLECTION PIPELINE
    # Generate random poses by applying Gaussian noise to a base pose(s)
    def generateGaussianNoisePoses(self, z_padding):
        """
        Generates a random end effector pose following a Gaussian distribution.

        @param z_padding: (float) slight padding to prevent the robot from colliding with the object when moving
                          to it via inverse kinematics
        
        @returns random_poses: filled array of generated end effector poses
        """
        
        # Base hand poses for different objects will vary as a result of manual trials on these objects.
        base_hand_poses = {
            "bottle": np.array([[0.1296842396259308, 0.014147371053695679, 0.13684210181236267, 0.09915781021118164, 1.520420789718628, 0.5952490568161011],
                [0.10374736785888672, -0.01886315643787384, 0.1315789520740509, 0.0, 1.553473711013794, 0.8928736448287964],
                [0.16033685207366943, 0.06602105498313904, 0.057894736528396606, 0.0, 1.570796251296997, 0.5952490568161011]]),
            "block": np.array([[0.0, 0.0, 0.2, 0.0, 1.570796251296997, 1.5707963705062866]]),
            "mug": np.array([[0.0, 0.0, 0.14210526645183563, 0.0, 1.570796251296997, 1.5707963705062866],
                [0.0, 0.002357885241508484, 0.17368420958518982, 0.0, 1.6195790767669678, 1.5707963705062866],
                [0.0, 0.0070736706256866455, 0.15789473056793213, 0.0, 1.6526315212249756, 1.5707963705062866]])
        }
        # Apply 6d gaussian noise to base hand pose
        sixd_noise = {
            "bottle": np.array([np.random.normal(0, 0.025, 1).item(), np.random.normal(0, 0.025, 1).item(), np.random.normal(0, 0.01, 1).item()+z_padding,
                0, np.random.normal(0, 0.01, 1).item(), np.random.normal(0, 0.01, 1).item()]),
            "block": np.array([0, np.random.normal(0, 0.04, 1).item(), np.random.normal(0, 0.01, 1).item()+z_padding, np.random.normal(0, 0.01, 1).item(),
                np.random.normal(0, 0.01, 1).item(), np.random.normal(0, 0.01, 1).item()]),
            "mug": np.array([np.random.normal(0, 0.01, 1).item(), np.random.normal(0, 0.01, 1).item(), np.random.normal(0, 0.01, 1).item()+z_padding,
                np.random.normal(0, 0.01, 1).item(), np.random.normal(0, 0.01, 1).item(), np.random.normal(0, 0.01, 1).item()])
        }
        noisy_poses = np.array([base_hand_poses[self.object_name][i] + sixd_noise[self.object_name] for i in range(len(base_hand_poses[self.object_name]))])
        return noisy_poses
    
    def tactileSanityCheck(self, depth, color):
        """
        Checks if the collected tactile data actually represent any grasps. Discards the data if the sensor
        doesn't pick up any readings.

        @param depth: (np.array) a pair of depth tactile readings from the tactile sensor; Shape: (2, 160, 120)
        @param color: (np.array) a pair of color tactile readings from the tactile sensor; Shape: (2, 160 , 120, 3)
        """

        if len(np.unique(depth[0], return_counts=True)[0]) == 1 or len(np.unique(depth[1], return_counts=True)[0]) == 1 or len(np.unique(color[0], return_counts=True)[0]) == 1 or len(np.unique(color[1], return_counts=True)[0]) == 1:
            print("Found invalid depth and color readings. Skipping this set of readings...")
            return None, None
        return depth, color
    
    def execute_pose(self, grasp_pose, velocity_scale, z_padding):
        """
        Executes a grasp given a specific hand pose
        """

        # 1. Move arm to pose and prepare gripper
        self.robot.manipulate_ee(grasp_pose, 'end', velocity_scale)
        self.robot.open_gripper()
        self.fixed_step_sim(500)

        # 2. Lower the arm by z=2
        lower_sixd_pose = grasp_pose.copy()
        lower_sixd_pose[2] = lower_sixd_pose[2] - z_padding
        self.robot.manipulate_ee(lower_sixd_pose, 'end', velocity_scale)
        self.fixed_step_sim(500)

        # 3. Close gripper to perform grasp
        self.robot.close_gripper()
        self.fixed_step_sim(500)

        # 4. Record depth and color
        self.digit_step()
        self.fixed_step_sim(200)

        # 5. Record tactile data
        color = np.asarray(self.color)      # (2, 160, 120, 3)
        depth = np.asarray(self.depth)      # (2, 160, 120)

        # 6. Make sure the recorded tactile data is valid
        depth, color = self.tactileSanityCheck(depth, color)

        # 7. Lift object for 5s to determine successful vs unsuccessful grasp
        upper_sixd_pose = grasp_pose.copy()
        upper_sixd_pose[2] += z_padding
        self.robot.manipulate_ee(upper_sixd_pose, 'end', velocity_scale)

        # 8. Record object z position to determine if it moved after a while
        grasped_object_z_pos = self.container.getPos()[2]
        self.fixed_step_sim(1500)

        # 9. Record success/failure
        final_object_z_pos = self.container.getPos()[2]

        # 10. Determine if the grabbed object stays in the same z position AND the z position is not 0 (as defined in container.getInitPos())
        delta_z = final_object_z_pos - grasped_object_z_pos
        grasp_outcome = True if delta_z > z_padding and final_object_z_pos > 0 else False

        return depth, color, grasp_outcome

    def collect_data(self):
        """
        This function executes a data collection loop by generating N gaussian-distributed end effector poses,
        then collecting the corresponding DIGIT tactile sensor readings on each finger of the gripper (as the 
        tactile data) as well as the end effector poses (as the visual data).

        The collected data is then used for stability classification in which the project aims to find the
        best representation of this data. This serves as the basis for further work on learning a generative
        model.
        """

        # Separate data arrays for tactile and visual data
        valid_random_poses = np.empty((0, 6))               # N trials x 6d pose
        depth_dataset = np.empty((0, 2, 160, 120))          # N trials x [Tactile data (depth) 160x120]
        color_dataset = np.empty((0, 2, 160, 120, 3))       # N trials x [Tactile data (color) 160x120x3]
        grasp_outcomes = np.empty((0))                      # N trials

        # Parameters of robot setup (can be changed)
        Z_PADDING = abs(0.2)        # Prevents the robot from colliding with the object when moving to it
        VELOCITY_SCALE = 0.15       # Scale up/down the robot movement speed

        count = 0
        success = 0
        failure = 0

        no_of_grasps = 200      # We need this number of success and failure grasps before stopping

        while success < no_of_grasps and failure < no_of_grasps:
            random_poses = self.generateGaussianNoisePoses(Z_PADDING)

            for random_pose in random_poses:
                print(f"Random pose {str(count+1)}: {random_pose}")

                # Execute grasp to get tactile readings and outcome
                depth, color, grasp_outcome = self.execute_pose(random_pose, VELOCITY_SCALE, Z_PADDING)

                # Only record the grasp data if the tactile data is valid
                if depth is None or color is None:
                    print(f"Not saving pose data to dataset.")
                else:
                    # Save recorded data to corresponding datasets
                    valid_random_poses = np.append(valid_random_poses, np.array([random_pose]), axis=0)
                    depth_dataset = np.append(depth_dataset, [depth], axis=0)
                    color_dataset = np.append(color_dataset, [color], axis=0)

                    if grasp_outcome:
                        np.append(grasp_outcomes, np.ones(shape=(1,)), axis=0)
                        success += 1
                    else:
                        np.append(grasp_outcomes, np.zeros(shape=(1,)), axis=0)
                        failure += 1
                    print(f"Successes: {success} | Failures: {failure}")

                # 7. Reset robot and arm only
                self.robot.reset()
                self.container.resetObject()
                self.fixed_step_sim(500)
                count += 1
        
        # Save collected data into .npy files for future loading
        self.save_dataset("depth_ds.npy", depth_dataset)
        self.save_dataset("color_ds.npy", color_dataset)
        self.save_dataset("poses_ds.npy", valid_random_poses)
        self.save_dataset("grasp_outcomes.npy", grasp_outcomes)

    def save_dataset(self, dataset_filename, dataset):
        CURR_DIR = os.getcwd()
        TARGET_DIR = "./src/baseline_model"

        dataset_path = os.path.join(CURR_DIR, TARGET_DIR, dataset_filename)
        np.save(dataset_path, dataset)
        print(f"Dataset saved to {dataset_path}")


    # HELPER FUNCTIONS FOR GENERATIVE MODEL
    def getRigidBodyDimensions(self, body_id):
        aabb_min, aabb_max = p.getAABB(body_id)
        width = aabb_max[0] - aabb_min[0]
        height = aabb_max[1] - aabb_min[1]
        depth = aabb_max[2] - aabb_min[2]
        return (width, height, depth)
    
    # Fetch the radius of a rigid body if it is p.GEOM_SPHERE or p.GEOM_CAPSULE
    def getRigidBodyRadius(self, body_id):
        obj_shape = p.getCollisionShapeData(body_id, -1)
        radius = None

        if obj_shape[0][1] == p.GEOM_SPHERE:
            radius = obj_shape[0][3][0]
        return radius if not None else None

    # Fetch the convexity of a rigid body
    # The 'convex' field contains a value between 0 and 1 that represents the convexity
    # of the shape, where 0 => perfectly convex and 1 => highly non-convex shape
    def getRigidBodyConvexity(self, body_id):
        # obj_shape = p.getCollisionShapeData(body_id, -1)
        # print(f"Object shape: {obj_shape}")
        # return obj_shape[0][2]['convex']
        return None

    # Calculate the curvature of a rigid body using the principal curvature estimation algorithm
    # This function only works for meshes that include a .obj file.
    def getRigidBodyCurvature(self, body_id):
        mesh_data = p.getMeshData(body_id)
        print(f"Object mesh data: {mesh_data}")
        
        # Convert the mesh data to a numpy array
        vertices, indices = np.array(mesh_data[0]), np.array(mesh_data[1])
        print(f"The object has {vertices} vertices.")
        return None

    def learning_framework(self):
        # 1. Collect a dataset of object poses and associated grasps. For each object pose, record the end effector position
        # orientation, and the gripper configuration (joint angles or width of gripper fingers)
        Z_PADDING = abs(0.2)
        random_poses_count = 1000

        # Block base hand poses
        positions = np.array([[0.0, 0.0, 0.18, 0.0, 1.570796251296997, 1.5707963705062866]])
        
        random_poses = self.generateGaussianNoisePoses(random_poses_count, positions, Z_PADDING)

        # 2. Train a Gaussian process generative model on the dataset to learn a mapping between the object poses and grasps.
        # Input: object pose (position + orientation of object)
        # Output: 
        obj_dims = self.getRigidBodyDimensions(self.container.ID)
        obj_radius = self.getRigidBodyRadius(self.container.ID)
        obj_convexity = self.getRigidBodyConvexity(self.container.ID)
        obj_curvature = self.getRigidBodyCurvature(self.container.ID)
        width, height, depth = obj_dims
        
        print(f"Width: {width} | Height: {height} | Depth: {depth} | Radius: {obj_radius} | Convexity: {obj_convexity}")    


    # CORE FUNCTIONS FOR RUNNING THE SIMULATION
    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
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
        self.readGenerativeModelButton()
        self.readOpenGripperButton()
        self.readCloseGripperButton()
        
        self.digit_step()
        self.fixed_step_sim(1000)
        
        return self.get_observation()

    def digit_step(self):
        self.color, self.depth = self.digits.render()
        self.digits.updateGUI(self.color, self.depth)
        
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
    
    # Reset user-defined parameters (GUI slider values)
    def resetUserDebugParameters(self):
        # Re-initialize sliders
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("Gripper opening length", 0, self.robot.gripper_range[1], self.robot.gripper_range[1])

        # Re-initialize simulation buttons
        self.resetSimulationButton = p.addUserDebugParameter("Reset simulation", 1, 0, 1)
        
        # Point cloud button and initial button values
        self.initButtonVals()
        self.pointCloudButton = p.addUserDebugParameter("Get point cloud", 1, 0, 1)

        # Button to get current joint coordinates and 6d pose of end effector
        self.jointObsButton = p.addUserDebugParameter("Get joint coordinates", 1, 0, 1)
        self.fetch6dButton = p.addUserDebugParameter("Get end effector pose", 1, 0 ,1)

        # Button for data collection and training approaches
        self.dataCollectionButton = p.addUserDebugParameter("Collect sensory data", 1, 0, 1)
        self.generativeModelButton = p.addUserDebugParameter("Execute generative model", 1, 0, 1)

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

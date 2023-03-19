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
from scipy.spatial import cKDTree



class ClutteredPushGrasp:
    # Global constants
    SIMULATION_STEP_DELAY = 1 / 250.


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
        self.robot.reset()

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

        # Load robot parameters
        self.Z_PADDING          = self.robot.robot_info["z_padding"]
        self.VELOCITY_SCALE     = self.robot.robot_info["velocity_scale"]

        # Load data collection parameters
        self.RANDOM_POSES_COUNT = self.robot.data_info["poses_count"]

        p.stepSimulation()

    # Hooks to pybullet.stepSimulation()
    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    # Auxiliary function for step_simulation with predetermined step size
    def fixed_step_sim(self, step_size):
        p.stepSimulation()
        for _ in range(step_size):
            self.step_simulation()

    # Initialize simulation GUI button values
    def initButtonVals(self):
        self.pointCloudButtonVal = 2.0
        self.jointObsButtonVal = 2.0
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
            self.collect_data_mlp()
        self.generativeModelButtonVal = p.readUserDebugParameter(self.generativeModelButton) + 1.0


    # HELPER FUNCTIONS FOR SIMULATION
    def gripperCollisionDetection(self):
        """
        Determines if the gripper is in contact with an object.
        """

        # linkIndex=13, 19 for DIGIT sensor on left and right finger respectively
        contact_points_left_sensor = p.getContactPoints(self.robot.id, self.container.ID, linkIndexA=13, linkIndexB=-1)
        contact_points_right_sensor = p.getContactPoints(self.robot.id, self.container.ID, linkIndexA=19, linkIndexB=-1)
        return len(contact_points_left_sensor+contact_points_right_sensor) > 0
    
    def checkTactileReadingsValid(self, depth, color):
        """
        Determines if the given depth and color tactile data provide any meaningful results, i.e. the mean is above a 
        certain threshold as a sanity check.

        We use a threshold of 1e-3 to determine the minimal acceptable mean for the tactile data
        """
        av_d0, av_d1 = np.mean(depth[0]), np.mean(depth[1])         # Mean of depth data for each finger
        av_c0, av_c1 = np.mean(color[0]), np.mean(color[1])         # Mean of color data for each finger
        
        # Additionally, we scale av_d0 and av_d1 to the range (0, 255) to match the range
        # of color data
        av_d0, av_d1 = av_d0 * 255, av_d1 * 255
        return all(i >= 1e-3 for i in [av_d0, av_d1, av_c0, av_c1])
    
    # HELPER FUNCTIONS FOR DATA COLLECTION PIPELINE
    # Generate random poses by applying Gaussian noise to a base pose(s)
    def generateGaussianNoisePose(self, pose, object_name):
        """
        Generates a random end effector pose following a Gaussian distribution.
        
        @returns random_poses: filled array of generated end effector poses
        """
        

        # Apply 6d gaussian noise to base hand pose
        sixd_noise = {
            # Baseline noise
            "block": np.random.normal(0, 0.01, 6),
            "bottle": np.random.normal(0, 0.01, 6),
            "sphere": np.random.normal(0, 0.01, 6),

            # MLP noise
            "block1": np.random.normal(0, 0.01, 6),
            "block2": np.random.normal(0, 0.01, 6),
            "block3": np.random.normal(0, 0.01, 6),
        }
        
        noisy_poses = pose + sixd_noise[object_name]
        noisy_poses[2] += self.Z_PADDING
        return noisy_poses
    
    def execute_pose(self, grasp_pose):
        """
        Executes a grasp given a specific hand pose

        @params grasp_pose: 6d numpy array of an end effector pose (cartesian coordinates (x, y, z) &
          euler angles (r, p, y))
        """

        # 1. Move arm to pose and prepare gripper
        print("Gripper: move to initial position for grasp")
        self.robot.manipulate_ee(grasp_pose, 'end', self.VELOCITY_SCALE)
        self.robot.open_gripper()
        self.fixed_step_sim(1000)

        # 2. Lower the arm by z=2
        lower_sixd_pose = grasp_pose.copy()
        lower_sixd_pose[2] -= self.Z_PADDING

        print('Gripper: lower gripper to object')
        self.robot.manipulate_ee(lower_sixd_pose, 'end', self.VELOCITY_SCALE)
        self.fixed_step_sim(1000)

        # 3. Close gripper to perform grasp
        print('Gripper: close gripper')
        self.robot.close_gripper()
        self.fixed_step_sim(200)

        # 4. Determine if gripper is in stable contact with object (measure over 500 steps)
        isGripperInContactTimeframe1 = self.gripperCollisionDetection()
        self.fixed_step_sim(500)
        isGripperInContactTimeframe2 = self.gripperCollisionDetection()

        # 5. Gripper contact is stable - record grasp and tactile data
        if isGripperInContactTimeframe1 and isGripperInContactTimeframe2:
            # 6. Record depth and color tactile data
            self.digit_step()
            self.fixed_step_sim(200)
            color = np.asarray(self.color)      # (2, 160, 120, 3)
            depth = np.asarray(self.depth)      # (2, 160, 120)

            # 7. Tactile data sanity check
            tactile_data_is_valid = self.checkTactileReadingsValid(depth, color)

            if tactile_data_is_valid:
                upper_grasp_pose = lower_sixd_pose.copy()
                upper_grasp_pose[2] += self.Z_PADDING
                
                # 7. Lift object for 5s to determine successful vs unsuccessful grasp
                print('Gripper: tactile data correct - lifting gripper...')
                self.robot.manipulate_ee(upper_grasp_pose, 'end', self.VELOCITY_SCALE)
                self.fixed_step_sim(200)

                # 8. Record object z position
                grasped_object_z_pos = self.container.getPos()[2]
                # 9. Record success/failure by determining if object moved after 1500 steps
                self.fixed_step_sim(1000)
                final_object_z_pos = self.container.getPos()[2]


                # 10. Determine if the grabbed object stays in the same z position AND the z position is not 0 (as defined in container.getInitPos())
                delta_z = abs(final_object_z_pos - grasped_object_z_pos)

                grasp_outcome = True if delta_z < self.Z_PADDING < final_object_z_pos else False

                return (depth, color, grasp_outcome)
            return None
        
        self.robot.reset()
        return None

    def save_dataset(self, dataset_filename, folder_name, dataset):
        CURR_DIR = os.getcwd()
        TARGET_DIR = "./src/" + folder_name

        dataset_path = os.path.join(CURR_DIR, TARGET_DIR, dataset_filename)
        np.save(dataset_path, dataset)
        print(f"Dataset saved to {dataset_path}")

    def collect_data(self):
        """
        This function executes a data collection loop by generating N gaussian-distributed end effector poses,
        then collecting the corresponding DIGIT tactile sensor readings on each finger of the gripper (as the 
        tactile data) as well as the end effector poses (as the visual data).

        The collected data is then used for stability classification in which the project aims to find the
        best representation of this data. This serves as the basis for further work on learning a generative
        model.
        """      

        # Arrays to populate when collecting data
        end_effector_poses = np.empty((self.RANDOM_POSES_COUNT, 6))                # End effector is a 6D structure
        tactile_depth_data = np.empty((self.RANDOM_POSES_COUNT, 2, 160, 120))      # Depth data (160x120) per finger (x2)
        tactile_color_data = np.empty((self.RANDOM_POSES_COUNT, 2, 160, 120, 3))   # Color data (160x120x3) per finger (x2)
        grasp_outcomes     = np.empty(0)

        # Counters for logging
        success_count = 0
        failure_count = 0

        # Base hand poses for different objects will vary as a result of manual trials on these objects.
        base_hand_poses = {
            "bottle": np.array([0.1296842396259308, 0.014147371053695679, 0.13684210181236267, 0.09915781021118164, 1.520420789718628, 0.5952490568161011]),
            "block1": np.array([0.0, 0.0, 0.2, 0.0, 1.570796251296997, 1.5707963705062866]),
            "sphere": np.array([0.0, 0.0, 0.2, 0.0, 1.570796251296997, 1.5707963705062866])
        }

        while success_count < self.RANDOM_POSES_COUNT or failure_count < self.RANDOM_POSES_COUNT:
            random_pose = self.generateGaussianNoisePose(base_hand_poses[self.object_name], self.object_name)
            generated_grasp_data = self.execute_pose(random_pose)

            if generated_grasp_data is not None:
                depth, color, grasp_is_good = generated_grasp_data

                # Only record the grasp data if the tactile data is valid
                if depth is None or color is None:
                    print(f"Not saving grasp data to dataset :(")
                else:
                    if grasp_is_good and (grasp_outcomes == 1).sum() < self.RANDOM_POSES_COUNT:
                        # Save recorded data to corresponding datasets
                        end_effector_poses = np.append(end_effector_poses, np.array([random_pose]), axis=0)
                        tactile_depth_data = np.append(tactile_depth_data, np.array([depth]), axis=0)
                        tactile_color_data = np.append(tactile_color_data, np.array([color]), axis=0)
                        grasp_outcomes = np.append(grasp_outcomes, np.ones(shape=(1,)), axis=0)
                        success_count += 1
                    elif not grasp_is_good and (grasp_outcomes == 0).sum() < self.RANDOM_POSES_COUNT:
                        # Save recorded data to corresponding datasets
                        end_effector_poses = np.append(end_effector_poses, np.array([random_pose]), axis=0)
                        tactile_depth_data = np.append(tactile_depth_data, np.array([depth]), axis=0)
                        tactile_color_data = np.append(tactile_color_data, np.array([color]), axis=0)
                        grasp_outcomes = np.append(grasp_outcomes, np.zeros(shape=(1,)), axis=0)
                        failure_count += 1
                    print(f"Data analysed and saved - Successes: {success_count} | Failures: {failure_count}")

            # 7. Reset robot and arm only
            self.robot.reset()
            self.container.resetObject()
            self.fixed_step_sim(500)

        folder_name = "baseline_model"
        # Save collected data into .npy files for future loading
        self.save_dataset("depth_ds.npy", folder_name, tactile_depth_data)
        self.save_dataset("color_ds.npy", folder_name, tactile_color_data)
        self.save_dataset("poses_ds.npy", folder_name, end_effector_poses)
        self.save_dataset("grasp_outcomes.npy", folder_name, grasp_outcomes)

    def collect_data_mlp(self):
        print("MLP data collection started...", flush=True)
        poses = self.load_mlp_poses()

        # Counters for logging
        success_count = 0
        failure_count = 0
        total_poses = 0

        SEED_POSE_COUNT = 10
        MLP_POSES_COUNT = 200
        TOTAL_POSES = MLP_POSES_COUNT * len(poses)

        end_effector_poses = np.empty((TOTAL_POSES, 6))                             # End effector is a 6D structure
        tactile_depth_data = np.empty((TOTAL_POSES, 2, 160, 120))                   # Depth data (160x120) per finger (x2)
        tactile_color_data = np.empty((TOTAL_POSES, 2, 160, 120, 3))                # Color data (160x120x3) per finger (x2)
        geometric_data     = self.getObjectGeometry(body_id=self.container.ID)      # W, H, D, convexity (3x2)
        grasp_outcomes     = np.empty(0)

        for seed_pose in poses[self.object_name]:
            print(f"Starting seed pose simulation: {seed_pose}")
            # We need 10 good and 10 bad grasps per seed pose
            while success_count < SEED_POSE_COUNT or failure_count < SEED_POSE_COUNT:
                noisy_pose = self.generateGaussianNoisePose(seed_pose, self.object_name)
                grasp_data = self.execute_pose(noisy_pose)

                if grasp_data is not None:
                    depth, color, grasp_is_good = grasp_data

                    # Record data if tactile data is valid
                    if depth is None or color is None:
                        print(f"Not saving grasp data to dataset :(")
                    else:
                        if grasp_is_good is True and success_count < SEED_POSE_COUNT:
                            # Save recorded data to corresponding datasets
                            end_effector_poses = np.append(end_effector_poses, np.array([noisy_pose]), axis=0)
                            tactile_depth_data = np.append(tactile_depth_data, np.array([depth]), axis=0)
                            tactile_color_data = np.append(tactile_color_data, np.array([color]), axis=0)
                            grasp_outcomes = np.append(grasp_outcomes, np.ones(shape=(1,)), axis=0)
                            success_count += 1
                            total_poses += 1
                            print(f"Data saved - Successes: {success_count} | Failures: {failure_count} | Total: {total_poses}")
                        elif grasp_is_good is False and failure_count < SEED_POSE_COUNT:
                            # Save recorded data to corresponding datasets
                            end_effector_poses = np.append(end_effector_poses, np.array([noisy_pose]), axis=0)
                            tactile_depth_data = np.append(tactile_depth_data, np.array([depth]), axis=0)
                            tactile_color_data = np.append(tactile_color_data, np.array([color]), axis=0)
                            grasp_outcomes = np.append(grasp_outcomes, np.zeros(shape=(1,)), axis=0)
                            failure_count += 1
                            total_poses += 1
                            print(f"Data saved - Successes: {success_count} | Failures: {failure_count} | Total: {total_poses}")
                        
                self.robot.reset_arm()
                self.container.resetObject()
                print('Gripper: resetted')
                self.fixed_step_sim(500)
            
            print("Collected enough data for seed pose. Moving to next pose")

            # Reset counters
            success_count = 0
            failure_count = 0

        # Save collected data into .npy files for future loading
        folder_name = "mlp_model"
        file_path = f"{self.object_name}_ds/"
        self.save_dataset(file_path + "depth_ds.npy", folder_name, tactile_depth_data)
        self.save_dataset(file_path + "color_ds.npy", folder_name, tactile_color_data)
        self.save_dataset(file_path + "poses_ds.npy", folder_name, end_effector_poses)
        self.save_dataset(file_path + "grasp_outcomes.npy", folder_name, grasp_outcomes)
            

    def load_mlp_poses(self):
        poses = {
            "block1": [
                (0.0, 0.0, 0.14210526645183563, 1.553473711013794, 1.553473711013794, 1.5707963705062866),
                (-0.13675791025161743, -0.011789470911026001, 0.05263157933950424, 0.0, 0.2644209861755371, 0.01653468608856201),
                (0.0, -0.12261052429676056, 0.09473684430122375, 0.0, 0.4957895278930664, 1.5707963705062866),
                (0.0, -0.14854736626148224, 0.08421052992343903, 0.0, 0.36357903480529785, 1.5707963705062866),
                (0.0, -0.016505271196365356, 0.16842105984687805, 0.0, 1.570796251296997, 1.5707963705062866),
                (-0.13675791025161743, -0.016505271196365356, 0.07894736528396606, 0.09915781021118164, 0.36357903480529785, 0.09920823574066162),
            ],
            "block2": [
                (0.0, -0.016505271196365356, 0.17894737422466278, 0.0, 1.570796251296997, 1.5707963705062866),
                (0.0, 0.014147371053695679, 0.17894737422466278, 0.0, 1.570796251296997, 1.5707963705062866),
                (0.0, -0.13204210996627808, 0.03684210404753685, 0.0, -0.033052682876586914, 1.5707963705062866),
                (0.0, -0.12261052429676056, 0.10526315867900848, 0.0, 0.4957895278930664, 1.5707963705062866),
                (-0.1414736807346344, 0.016505271196365356, 0.07894736528396606, -3.140000104904175, 0.2644209861755371, -0.06613874435424805),
                (-0.11317894607782364, 0.014147371053695679, 0.10526315867900848, -3.140000104904175, 0.6941053867340088, -0.06613874435424805),
                (-0.002357885241508484, 0.11553683876991272, 0.12105263024568558, -3.140000104904175, 0.7271578311920166, -1.5707963705062866),
                (0.20749473571777344, 0.13911578059196472, 0.031578946858644485, 3.140000104904175, -3.140000104904175, -1.5707963705062866),
            ],
            "block3": [
                (-0.007073685526847839, 0.0, 0.16842105984687805, 0.0, 1.570796251296997, 0.0),
                (0.016505271196365356, 0.0, 0.1631578952074051, 0.0, 1.570796251296997, 0.0),
                (0.06837895512580872, 0.0, 0.15263158082962036, 0.09915781021118164, 2.0823161602020264, 0.0),
                (0.13440001010894775, -0.007073685526847839, 0.021052632480859756, 3.140000104904175, 3.140000104904175, -0.033069491386413574),
                (-0.1296842098236084, 0.0070736706256866455, 0.09473684430122375, -0.09915804862976074, 0.4627368450164795, -0.049604058265686035),
                (-0.1414736807346344, 0.0070736706256866455, 0.06842105090618134, -0.09915804862976074, 0.4627368450164795, -0.049604058265686035),
                (0.0, -0.10138948261737823, 0.08947368711233139, 3.140000104904175, 0.7602105140686035, 1.5707963705062866),
                (0.0, 0.0, 0.15789473056793213, 0.0, 1.570796251296997, 1.5707963705062866),
                (0.0, -0.12496842443943024, 0.10000000149011612, 0.0, 0.5288419723510742, 1.5707963705062866),
            ],
        }
        return poses


    # HELPER FUNCTIONS FOR GENERATIVE MODEL
    
    # Fetch the radius of a rigid body if it is p.GEOM_SPHERE or p.GEOM_CAPSULE
    def getRigidBodyRadius(self, body_id):
        obj_shape = p.getCollisionShapeData(body_id, -1)
        radius = None

        if obj_shape[0][1] == p.GEOM_SPHERE:
            radius = obj_shape[0][3][0]
        return radius if not None else None

    # Calculate the curvature of a rigid body using the principal curvature estimation algorithm
    # This function only works for meshes that include a .obj file.
    def getRigidBodyCurvature(self, body_id, k):
        mesh_data = p.getMeshData(body_id)

        # Convert the mesh data to a numpy array
        object_vertices, object_indices = np.array(mesh_data[0]), np.array(mesh_data[1])

        if object_vertices > 0:
            # Compute the principal curvatures at each vertex
            curvatures = np.zeros((len(object_indices), 2))
            tree = cKDTree(object_indices)
            for i, vertex in enumerate(object_indices):
                # Find the 10 nearest neighbors to the current vertex
                _, indices = tree.query(vertex, k=10)
                neighbors = object_indices[indices, :]

                # Compute the covariance matrix of the neighbors
                centroid = np.mean(neighbors, axis=0)
                centered = neighbors - centroid
                cov = np.dot(centered.T, centered)

                # Compute the eigenvalues and eigenvectors of the covariance matrix
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                # Sort the eigenvalues in descending order
                indices = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[indices]
                eigenvectors = eigenvectors[:, indices]

                # Compute the principal curvatures as the reciprocals of the eigenvalues
                curvatures[i] = np.abs(1.0 / eigenvalues[:2])
            
            # Get the top k eigenvalues
            return curvatures[:k]
        else:
            return np.zeros((k, 2))

    # Get geometric features of an object
    def getObjectGeometry(self, body_id):
        obj_shape = p.getCollisionShapeData(objectUniqueId=body_id, linkIndex=-1)
        width, depth, height = obj_shape[0][3]
        curvature_data = self.getRigidBodyCurvature(body_id, k=3)
        curvature_data = curvature_data.flatten()
        return (width, depth, height, curvature_data)


    # CORE FUNCTIONS FOR RUNNING THE SIMULATION
    def step(self):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """

        # in the step simulation, get the joint orientation
        sixd = self.robot.get_joint_obs()

        # in the step simulation, read the point cloud
        self.readPointCloudButton()
        self.readResetSimulationButton()
        self.readJointObsButton(sixd)
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
        # Re-initialize simulation buttons
        self.initButtonVals()
        self.resetSimulationButton  = p.addUserDebugParameter("Reset simulation", 1, 0, 1)
        self.pointCloudButton       = p.addUserDebugParameter("Get point cloud", 1, 0, 1)
        self.jointObsButton         = p.addUserDebugParameter("Get joint coordinates", 1, 0, 1)
        self.dataCollectionButton   = p.addUserDebugParameter("Collect sensory data", 1, 0, 1)
        self.generativeModelButton  = p.addUserDebugParameter("Execute generative model", 1, 0, 1)
        self.openGripperButton      = p.addUserDebugParameter("Open gripper", 1, 0, 1)
        self.closeGripperButton     = p.addUserDebugParameter("Close gripper", 1, 0, 1)

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

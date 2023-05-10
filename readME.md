<h1 align="center">Robust Robotic Grasping Utilising Touch Sensing</h1>
<p align="center">UCL COMP0029 Final Year Project Repository</p>

<p>This repository contains all the code and documentation mentioned in my FYP project report. The code has only been tested on a Ubuntu 20.04 dual-boot set-up. It is not guaranteed to work perfectly on other Linux distributions.</p>

<img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black"/><img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white"/>

https://user-images.githubusercontent.com/52330996/216103556-56d965ef-75c2-4b0a-8596-9a44c21b57e2.mp4

<hr>

## Prerequisites
Before compiling and testing the code from this repository, please ensure that your system has been properly configured with the following requirements:
1. Linux Operating System (preferably Ubuntu 20.04). The code for this project was only developed and tested on Ubuntu 20.04. It is not guaranteed to compile and run fully functionally on other operating system and devices.
2. Python 3.8.16

## Deployment
1. Clone the repository or download a zip file containing the repository code.
2. Create a Python virtual environment with Python 3.8.16. Ensure that the virtual environment is created in the root directory of the project:
3. Activate the virtual environment:
   ```
   source fyp_env/activate
   ```
4. Install `PyTorch`:
   ```
   pip3 install torch==1.13.1+cu116 torchaudio==0.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
   ```
5. Install all required Python packages:
   ```
   pip install -r requirements.txt
   ```
6. Run the Pybullet simulation:
   ```
   python src/main.py
   ```

## Using the simulation
There are several core functionalities provided by our Pybullet simulation. These functionalities are listed as either sliders or buttons on the right panel of the simulation GUI.

- **Manual manipulation of the robot**: Adjust the six available sliders for the end effector pose (x,y,z,roll,pitch,yaw) and the slider for the gripper length (0 = gripper is completely closed, maximum = gripper is completely open).
- **Reset simulation**: Resets all sliders and the robot (end effector position).
- **Get joint coordinates**: Retrieves all movable joints of the robot arm and gripper.
- **Get object features**: Retrieves the object features of the created object mesh in the simulation. Currently this function returns the 3 principal curvatures of the object, one for each dimension (width, height, depth).
- **Collect data (baseline)**: Data collection pipeline to collect data for the baseline approach. *To run this function, please ensure the `object_name` property is set to `block` in the `parameters.yaml` file*. This function will not work for any other object name since manually-selected end effector poses were only collected for `block`.
- **Collect data (proposed)**: Data collection pipeline to collect data for the proposed approach (MLP). *To run this function, please ensure the `object_name` property is set to one of the following in the `parameters.yaml` file*:
  ```
  object_name: block1/block2/block3/bottle1/bottle2/bottle3/cylinder1/cylinder2/cylinder3
  ```

## Training models
The baseline and proposed models for this project are trained separately from the simulation code since it requires a significantly larger amount of computing power. The relevant code can be found in the Jupyter notebooks (with the `.ipynb` file extension) in the `models/baseline_model` or `models/mlp_model` directories. If this code needs to be assessed, please ensure that these notebooks are run with `cuda` enabled. Otherwise, there is a risk of insufficient memory allocation to training the models.

## Credits
This simulation was developed collaboratively with Jeffery Wei, a Master's 23' project student working with the Statistical Machine Learning Group at UCL.

Part of the code in this repository is taken from the following repositories:
- [https://github.com/yaseminb/robot_simulator](https://github.com/yaseminb/robot_simulator)
- [https://github.com/ElectronicElephant/pybullet_ur5_robotiq](https://github.com/ElectronicElephant/pybullet_ur5_robotiq)
<h1 align="center">UCL-FYP</h1>
<p align="center">Final year project repository on Robust Robotic Grasping Utilising Touch Sensing</p>

<hr>

This repository is split into several parts:
1. Pybullet simulation

## Prerequisites
Please ensure that you are using the following versions of software and tools:
- Python: v3.9
- Python `virtualenv` installed on your OS
- Linux OS (preferrably Ubuntu 20.04)

## Pybullet simulation
The Pybullet simulation for this project utilises a Panda arm and a <a href="https://github.com/a-price/robotiq_arg85_description">Robotiq-ARG85</a> gripper. The relevant code can be found in the `src/sim` directory.

#### Simulation installation
Follow the following steps to set up the Pybullet simulation:
1. Install Python 3.9
   ```
   sudo apt-get install python3.9
   ```
2. Find the path where Python3.9 in installed
   ```
   which python3.9
   ```
3. Install `python3.9-dev` (for installing legacy packages)
   ```
   sudo apt install python3.9-dev
   ```
4. Create a new virtual environment using `virtualenv` and the specific version of Python
   ```
   virtualenv <virtual-environment-name> --python="path/to/python3.9"
   ```
5. Activate the virtual environment
   ```
   source <virtual-environment-name>/bin/activate
   ```
6. Install required packages for the simulator
   ```
   pip install -r requirements.txt
   ```

#### Running the simulator
1. Navigate to the `RobotSimulator.py` file
2. Run the command in the terminal:
   ```
   python RobotSimulator.py
   ```
3. Confirm the `.yaml` file parameters by clicking `Enter`

#### Potential errors when running `RobotSimulator.py`
1. If you get a `DQ_SerialManipulator: No constructor defined!` error, there is a version mismatch in `dqrobotics` that affects the code. To solve this issue, please downgrade the `dqrobotics` package:
   ```
   pip install dqrobotics==20.4.0.31
   ```
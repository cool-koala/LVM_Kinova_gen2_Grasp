# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a ROS-based robotic grasping system that integrates:
- **Kinova Gen2 robot arm** (j2s6s200 model) with 2-finger gripper and ROS drivers
- **RealSense depth camera** for perception
- **Deep learning grasp detection** using SGDN neural network
- **Vision-Language Model integration** for voice-controlled grasping via Qwen-VL-Max

### Key Components

1. **ROS Workspace** (`Kinova_ws/`):
   - `kinova-ros/`: Official Kinova ROS packages for arm control
   - `sim_grasp/`: Custom grasp detection and execution package
   - `realsense-ros/`: RealSense camera drivers

2. **Vision-Language Interface** (`qianwenVLmax_api/`):
   - Voice-controlled grasping using Qwen-VL-Max multimodal LLM
   - Speech recognition and text-to-speech capabilities
   - Object detection and coordinate extraction from camera feeds

3. **Grasp Detection Pipeline**:
   - Deep learning model (SGDN) for grasp pose estimation
   - Collision detection and depth analysis
   - Coordinate transformation from camera to robot base frame

## Common Commands

### Build the ROS workspace:
```bash
cd Kinova_ws
catkin_make
source devel/setup.bash
```

### Launch the complete grasping system:
```bash
# Launch robot arm, camera, and grasp detection
roslaunch sim_grasp sim_grasp_kinova.launch

# Alternative: Launch just the robot arm
roslaunch kinova_bringup kinova_robot.launch
```

### Run individual components:
```bash
# Start grasp detection node
rosrun sim_grasp grasp_detect.py

# Start vision-language interface
cd qianwenVLmax_api
python qinwen_Grasp_advaced.py
```

### Test grasp detection:
```bash
# Trigger grasp detection
rostopic pub -1 /grasp/grasp_detect_run std_msgs/Int8 'data: 0'

# Capture image for vision model
rostopic pub -1 /Capture_picture std_msgs/String 'data: 'your''
```

## System Configuration

### Robot Configuration:
- Default robot type: `j2s6s200` (6-DOF arm with 2-finger gripper)
- Control modes: Position control, trajectory following
- Coordinate frames: Base frame to end-effector transformation

### Camera Setup:
- RealSense D435/D455 depth camera
- Calibrated camera-to-end-effector transformation
- Aligned depth and RGB image streams

### Neural Network:
- SGDN model for grasp detection
- Model checkpoint: `sim_grasp/scripts/ckpt/epoch_0064_acc_0.1957_.pth`
- Input: Depth images, Output: Grasp poses (position, angle, width, confidence)

## Key File Locations

- **Launch files**: `Kinova_ws/src/kinova-ros/sim_grasp/launch/`
- **Grasp detection**: `Kinova_ws/src/kinova-ros/sim_grasp/scripts/grasp_detect.py`
- **Robot control**: `Kinova_ws/src/kinova-ros/sim_grasp/src/sim_arm_grasp_kinova.cpp`
- **VLM interface**: `qianwenVLmax_api/qinwen_Grasp_advaced.py`
- **Model weights**: `Kinova_ws/src/kinova-ros/sim_grasp/scripts/ckpt/`

## Important Notes

- The system requires proper camera calibration for accurate grasp execution
- Voice commands are processed through Qwen-VL-Max API (requires API key configuration)
- Shared network storage is used for image capture (IP address needs to be configured based on your network setup)
- A Python script monitor (`python_script_monitor.py`) watches for generated scripts and executes them to publish ROS topics
- The robot workspace and safety limits are configured in launch files
- Always ensure the robot is in a safe position before running autonomous grasping

## Additional Components

### Python Script Monitor
- **File**: `python_script_monitor.py`
- **Purpose**: Monitors shared directory for generated Python scripts from VLM interface
- **Function**: Automatically executes scripts that publish ROS topics for robot control
- **Usage**: Run as a separate process alongside the main system
- **Configuration**: Modify directory path and target filenames as needed
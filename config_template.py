#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Template for LVM_Kinova_gen2_Grasp
配置模板文件

Copy this file to config.py and fill in your actual values.
复制此文件为config.py并填入您的实际配置值。

Usage:
1. Copy: cp config_template.py config.py
2. Edit config.py with your actual credentials and paths
3. Import in your scripts: from config import *
"""

# ============= API Keys / API密钥 =============
# Qwen-VL-Max API Configuration
QWEN_API_KEY_FILE = 'path/to/your/qwenapi.txt'

# Huawei Cloud Configuration
HUAWEI_CLOUD_AK = "YOUR_HUAWEI_CLOUD_ACCESS_KEY"
HUAWEI_CLOUD_SK = "YOUR_HUAWEI_CLOUD_SECRET_KEY" 
HUAWEI_CLOUD_PROJECT_ID = "YOUR_PROJECT_ID"
HUAWEI_CLOUD_REGION = "cn-east-3"  # Change to your region

# Baidu API Configuration
BAIDU_API_KEY = "YOUR_BAIDU_API_KEY"
BAIDU_SECRET_KEY = "YOUR_BAIDU_SECRET_KEY"

# OpenAI API Configuration (for old_grasp.py and gpt4v2.py)
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# ============= Network Configuration / 网络配置 =============
# Network storage IP address (replace with your actual IP)
NETWORK_STORAGE_IP = "192.168.1.100"  # Replace with your network IP
CAMERA_CAPTURE_SHARE = f"\\\\{NETWORK_STORAGE_IP}\\Camera_capture"
CAMERA_CAPTURE_LOCAL = f"/path/to/your/Camera_capture"

# Image paths for VLM processing
RGB_IMAGE_NETWORK_PATH = f"file:////{NETWORK_STORAGE_IP}/Camera_capture/rgb.png"
RGB_IMAGE_LOCAL_PATH = "/path/to/your/Camera_capture/rgb.png"

# ============= Model Configuration / 模型配置 =============
# SGDN model path
GRASP_MODEL_PATH = "/path/to/your/workspace/src/kinova-ros/sim_grasp/scripts/ckpt/epoch_0064_acc_0.1957_.pth"

# Device configuration
DEVICE_NAME = "cpu"  # Change to "cuda:0" if you have GPU

# Model parameters
ANGLE_CLASSES = 18
GRASP_WIDTH_MAX = 0.1  # meters

# ============= Robot Configuration / 机器人配置 =============
# Robot type
ROBOT_TYPE = "j2s6s200"

# Camera to robot transformation (adjust based on your calibration)
CAMERA_END_TRANSLATION = [0.0259, 0.0772, -0.1663]  # [X, Y, Z] in meters

# Robot initial pose
ARM_INIT_POSE = [0, -0.3, 0.318, 3.14159, 0, 0]  # [X, Y, Z, R, P, Y]

# ============= Voice Configuration / 语音配置 =============
# Voice synthesis settings
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

# Wake word detection
WAKE_WORDS = ["小海", "小孩"]

# ============= Directory Monitoring / 目录监控 =============
# Python script monitor configuration
MONITOR_DIRECTORY = "/home/your_user/Camera_capture"  # Local monitoring directory
TARGET_SCRIPT_FILES = ['generated.py', 'genzerated.py']

# ============= Safety Configuration / 安全配置 =============
# Collision detection parameters
FINGER_LENGTH_1 = 0.02  # meters
FINGER_LENGTH_2 = 0.005  # meters

# Grasp detection thresholds
CONFIDENCE_THRESHOLD = 0.3
PEAK_DISTANCE = 1

# ============= Usage Instructions / 使用说明 =============
"""
使用说明 / Usage Instructions:

1. 复制此文件 / Copy this file:
   cp config_template.py config.py

2. 编辑config.py填入您的实际值 / Edit config.py with your actual values

3. 在脚本中导入配置 / Import in your scripts:
   from config import *

4. 确保所有API密钥有效 / Ensure all API keys are valid

5. 根据您的网络设置调整IP地址 / Adjust IP addresses for your network

6. 根据您的机器人标定调整变换参数 / Adjust transformation parameters based on calibration

安全提醒 / Security Reminder:
- 永远不要提交包含真实API密钥的config.py文件
- Never commit config.py file with real API keys
- 将config.py添加到.gitignore文件中
- Add config.py to your .gitignore file
"""
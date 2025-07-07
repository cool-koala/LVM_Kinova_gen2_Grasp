# LVM_Kinova_gen2_Grasp

[English](#english) | [中文](#中文)

---

## English

### Overview

LVM_Kinova_gen2_Grasp is an intelligent robotic grasping system that integrates Vision-Language Models (VLM) with robotic manipulation. The system combines voice commands, visual perception, and deep learning to enable natural language-controlled robotic grasping using a Kinova Gen2 robot arm.

### Key Features

- **Voice-Controlled Grasping**: Natural language commands processed through Qwen-VL-Max multimodal LLM
- **Visual Perception**: RealSense depth camera for 3D scene understanding
- **Deep Learning Grasp Detection**: SGDN neural network for optimal grasp pose estimation
- **Real-time Processing**: Integrated pipeline from voice command to robot execution
- **Safety Features**: Collision detection and workspace monitoring

### System Architecture

```
Voice Command → Speech Recognition → Vision-Language Model → Object Detection → 
Grasp Planning → Robot Control → Execution Feedback
```

### Hardware Requirements

- **Robot**: Kinova Gen2 j2s6s200 (6-DOF arm with 2-finger gripper)
- **Camera**: Intel RealSense D435/D455 depth camera
- **Computer**: Ubuntu 18.04/20.04 with ROS Melodic/Noetic
- **Network**: Shared storage for image capture and processing

### Software Dependencies

- **ROS**: Robot Operating System (Melodic/Noetic)
- **Python**: 3.6+ with deep learning libraries
- **Vision-Language Model**: Qwen-VL-Max API
- **Speech Processing**: Huawei Cloud Speech services
- **Deep Learning**: PyTorch for grasp detection
- **Grasp Detection Model**: GGCNN2 (Generative Grasping CNN v2)

See `requirements.txt` for complete dependency list.

### Grasp Detection Model

The system uses **GGCNN2 (Generative Grasping CNN v2)**, a lightweight convolutional neural network for robotic grasp detection:

#### Model Architecture
- **Base Network**: GGCNN2 with 18-class angle prediction
- **Input**: Single-channel depth images (480×480 pixels)
- **Output**: Three prediction maps:
  - **Grasp Quality**: Confidence score for each pixel
  - **Grasp Angle**: 18-class discretized grasp angles (0-π radians)
  - **Grasp Width**: Optimal gripper opening width (0-0.1 meters)

#### Technical Details
- **Framework**: PyTorch implementation
- **Architecture**: Encoder-decoder with dilated convolutions
- **Training**: Pre-trained on grasp datasets
- **Inference Time**: ~50ms per frame on CPU
- **Model Size**: Lightweight for real-time applications

#### Prediction Pipeline
1. **Preprocessing**: Depth image inpainting and normalization
2. **Neural Network**: GGCNN2 forward pass
3. **Post-processing**: Peak detection and coordinate transformation
4. **Output**: Ranked list of grasp candidates with [x, y, angle, width, confidence]

### Installation

1. **Install ROS**:
   ```bash
   # For Ubuntu 18.04
   sudo apt install ros-melodic-desktop-full
   # For Ubuntu 20.04
   sudo apt install ros-noetic-desktop-full
   ```

2. **Clone and build the workspace**:
   ```bash
   git clone <repository-url>
   cd LVM_Kinova_gen2_Grasp/Kinova_ws
   catkin_make
   source devel/setup.bash
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   - Set up Qwen-VL-Max API key in `qianwenVLmax_api/qwenapi.txt`
   - Configure Huawei Cloud credentials (AK/SK) in `qinwen_Grasp_advaced.py`
   - Set Baidu API credentials for speech services
   - **Note**: All API keys have been removed from code for security. Please add your own keys before use.

### Usage

1. **Start the complete system**:
   ```bash
   roslaunch sim_grasp sim_grasp_kinova.launch
   ```

2. **Start the voice interface**:
   ```bash
   cd qianwenVLmax_api
   python qinwen_Grasp_advaced.py
   ```

3. **Start the Python script monitor**:
   ```bash
   python python_script_monitor.py
   ```

4. **Give voice commands**:
   - Wake word: "小海" (Xiaohai)
   - Example: "小海，帮我拿那个苹果" (Xiaohai, help me get that apple)

### System Workflow

1. **Voice Input**: User speaks natural language command
2. **Speech Recognition**: Convert speech to text using Huawei Cloud services
3. **Vision Capture**: Take photo with RealSense camera
4. **VLM Processing**: Qwen-VL-Max identifies target object and location
5. **Grasp Planning**: SGDN network calculates optimal grasp pose
6. **Robot Execution**: Kinova arm executes the grasp
7. **Feedback**: Audio confirmation of action completion

### Configuration

- **Robot Parameters**: Modify `kinova_bringup/launch/config/robot_parameters.yaml`
- **Camera Calibration**: Update camera-to-robot transformation in source files
- **Network Setup**: Configure IP addresses for shared storage access
- **API Keys**: Set up cloud service credentials

### File Structure

```
LVM_Kinova_gen2_Grasp/
├── Kinova_ws/                 # ROS workspace
│   └── src/
│       ├── kinova-ros/        # Kinova robot drivers
│       ├── sim_grasp/         # Grasp detection package
│       └── realsense-ros/     # Camera drivers
├── qianwenVLmax_api/         # Vision-Language interface
├── python_script_monitor.py  # Script execution monitor
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── CLAUDE.md                 # Development guidance
```

### Troubleshooting

- **Robot Connection**: Ensure USB connection and proper permissions
- **Camera Issues**: Check RealSense drivers and USB 3.0 connection
- **Network Problems**: Verify shared storage accessibility
- **API Errors**: Check internet connection and API key validity

### Security and Configuration

#### API Key Management
- **Security**: All API keys have been removed from the codebase for security
- **Configuration**: Use `config_template.py` as a template:
  ```bash
  cp config_template.py config.py
  # Edit config.py with your actual credentials
  ```
- **Git Ignore**: The `.gitignore` file prevents accidental commit of sensitive data
- **Best Practice**: Never commit real API keys or credentials to version control

#### Network Configuration
- Replace placeholder IP addresses with your actual network configuration
- Update file paths to match your system setup
- Configure camera calibration parameters for your specific setup

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Note**: Ensure no sensitive information (API keys, credentials, IP addresses) is included in commits.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 中文

### 概述

LVM_Kinova_gen2_Grasp 是一个智能机器人抓取系统，将视觉-语言模型(VLM)与机器人操作相结合。该系统结合语音命令、视觉感知和深度学习，通过 Kinova Gen2 机械臂实现自然语言控制的机器人抓取。

### 主要特性

- **语音控制抓取**：通过千问-VL-Max多模态大语言模型处理自然语言命令
- **视觉感知**：RealSense深度相机进行3D场景理解
- **深度学习抓取检测**：SGDN神经网络进行最优抓取姿态估计
- **实时处理**：从语音命令到机器人执行的集成管道
- **安全特性**：碰撞检测和工作空间监控

### 系统架构

```
语音命令 → 语音识别 → 视觉-语言模型 → 目标检测 → 
抓取规划 → 机器人控制 → 执行反馈
```

### 硬件要求

- **机器人**：Kinova Gen2 j2s6s200（6自由度机械臂，2指夹爪）
- **相机**：Intel RealSense D435/D455深度相机
- **计算机**：Ubuntu 18.04/20.04 with ROS Melodic/Noetic
- **网络**：用于图像捕获和处理的共享存储

### 软件依赖

- **ROS**：机器人操作系统（Melodic/Noetic）
- **Python**：3.6+及深度学习库
- **视觉-语言模型**：千问-VL-Max API
- **语音处理**：华为云语音服务
- **深度学习**：PyTorch用于抓取检测
- **抓取检测模型**：GGCNN2（生成式抓取CNN v2）

完整依赖列表请参考 `requirements.txt`。

### 抓取检测模型

系统使用 **GGCNN2（生成式抓取卷积神经网络 v2）**，这是一个用于机器人抓取检测的轻量级卷积神经网络：

#### 模型架构
- **基础网络**：带有18类角度预测的GGCNN2
- **输入**：单通道深度图像（480×480像素）
- **输出**：三个预测图：
  - **抓取质量**：每个像素的置信度分数
  - **抓取角度**：18类离散化抓取角度（0-π弧度）
  - **抓取宽度**：最优夹爪开合宽度（0-0.1米）

#### 技术细节
- **框架**：PyTorch实现
- **架构**：带有膨胀卷积的编码器-解码器
- **训练**：在抓取数据集上预训练
- **推理时间**：CPU上约50ms每帧
- **模型大小**：轻量级设计适合实时应用

#### 预测流程
1. **预处理**：深度图像修复和归一化
2. **神经网络**：GGCNN2前向传播
3. **后处理**：峰值检测和坐标变换
4. **输出**：按[x, y, 角度, 宽度, 置信度]排序的抓取候选列表

### 安装步骤

1. **安装ROS**：
   ```bash
   # Ubuntu 18.04
   sudo apt install ros-melodic-desktop-full
   # Ubuntu 20.04
   sudo apt install ros-noetic-desktop-full
   ```

2. **克隆并构建工作空间**：
   ```bash
   git clone <repository-url>
   cd LVM_Kinova_gen2_Grasp/Kinova_ws
   catkin_make
   source devel/setup.bash
   ```

3. **安装Python依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **配置API密钥**：
   - 在 `qianwenVLmax_api/qwenapi.txt` 中设置千问-VL-Max API密钥
   - 在 `qinwen_Grasp_advaced.py` 中配置华为云凭据（AK/SK）
   - 设置百度API凭据用于语音服务
   - **注意**：为了安全考虑，所有API密钥已从代码中移除。请在使用前添加您自己的密钥。

### 使用方法

1. **启动完整系统**：
   ```bash
   roslaunch sim_grasp sim_grasp_kinova.launch
   ```

2. **启动语音界面**：
   ```bash
   cd qianwenVLmax_api
   python qinwen_Grasp_advaced.py
   ```

3. **启动Python脚本监视器**：
   ```bash
   python python_script_monitor.py
   ```

4. **发出语音命令**：
   - 唤醒词："小海"
   - 示例："小海，帮我拿那个苹果"

### 系统工作流程

1. **语音输入**：用户说出自然语言命令
2. **语音识别**：使用华为云服务将语音转换为文本
3. **视觉捕获**：使用RealSense相机拍照
4. **VLM处理**：千问-VL-Max识别目标物体和位置
5. **抓取规划**：SGDN网络计算最优抓取姿态
6. **机器人执行**：Kinova机械臂执行抓取
7. **反馈**：语音确认动作完成

### 配置说明

- **机器人参数**：修改 `kinova_bringup/launch/config/robot_parameters.yaml`
- **相机标定**：更新源文件中的相机到机器人变换
- **网络设置**：配置共享存储访问的IP地址
- **API密钥**：设置云服务凭据

### 文件结构

```
LVM_Kinova_gen2_Grasp/
├── Kinova_ws/                 # ROS工作空间
│   └── src/
│       ├── kinova-ros/        # Kinova机器人驱动
│       ├── sim_grasp/         # 抓取检测包
│       └── realsense-ros/     # 相机驱动
├── qianwenVLmax_api/         # 视觉-语言接口
├── python_script_monitor.py  # 脚本执行监视器
├── requirements.txt          # Python依赖
├── README.md                 # 本文件
└── CLAUDE.md                 # 开发指南
```

### 故障排除

- **机器人连接**：确保USB连接和正确的权限
- **相机问题**：检查RealSense驱动和USB 3.0连接
- **网络问题**：验证共享存储的可访问性
- **API错误**：检查网络连接和API密钥有效性

### 安全性和配置

#### API密钥管理
- **安全性**：为了安全考虑，所有API密钥已从代码库中移除
- **配置**：使用 `config_template.py` 作为模板：
  ```bash
  cp config_template.py config.py
  # 编辑config.py填入您的实际凭据
  ```
- **Git忽略**：`.gitignore` 文件防止意外提交敏感数据
- **最佳实践**：永远不要将真实的API密钥或凭据提交到版本控制

#### 网络配置
- 将占位符IP地址替换为您的实际网络配置
- 更新文件路径以匹配您的系统设置
- 为您的特定设置配置相机标定参数

### 贡献

1. Fork 仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

**注意**：确保提交中不包含敏感信息（API密钥、凭据、IP地址）。

### 许可证

本项目采用MIT许可证 - 详情请参阅LICENSE文件。

---

### System Pipeline Diagram / 系统流程图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Voice Input   │────▶│ Speech Recognition│────▶│  Wake Word Det. │
│   语音输入      │     │    语音识别     │     │   唤醒词检测    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Robot Control  │◀────│  Grasp Planning │◀────│  Camera Capture │
│   机器人控制    │     │    抓取规划     │     │    相机捕获     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
          │                       ▲                       │
          │                       │                       ▼
          ▼                       │               ┌─────────────────┐
┌─────────────────┐               │               │ VLM Processing  │
│ Execution Feed. │               │               │ 视觉语言模型处理│
│   执行反馈      │               │               └─────────────────┘
└─────────────────┘               │                       │
                                  │                       ▼
                                  │               ┌─────────────────┐
                                  │               │ Object Detection│
                                  │               │   目标检测      │
                                  │               └─────────────────┘
                                  │                       │
                                  │                       ▼
                                  │               ┌─────────────────┐
                                  └───────────────│ Grasp Detection │
                                                  │   抓取检测      │
                                                  └─────────────────┘
```

### Component Interaction / 组件交互

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             ROS Environment / ROS环境                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  Kinova Driver  │  │ RealSense Driver│  │ Grasp Detection │                 │
│  │   Kinova驱动    │  │   相机驱动      │  │   抓取检测      │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ ROS Topics / ROS话题
                                  │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Python Script Monitor / Python脚本监视器                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  File Watcher   │  │ Script Executor │  │ ROS Publisher   │                 │
│  │   文件监视器    │  │   脚本执行器    │  │   ROS发布器     │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ Generated Scripts / 生成的脚本
                                  │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        VLM Interface / 视觉语言模型接口                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Speech Recog.   │  │  Qwen-VL-Max   │  │ Text-to-Speech  │                 │
│  │   语音识别      │  │   千问VL模型    │  │   语音合成      │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```
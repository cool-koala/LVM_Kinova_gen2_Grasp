#include "kinova_driver/kinova_api.h"
#include "kinova_driver/kinova_arm.h"
#include <kinova_driver/kinova_ros_types.h>
#include "kinova_driver/kinova_tool_pose_action.h"
#include "kinova_driver/kinova_joint_angles_action.h"
#include "kinova_driver/kinova_fingers_action.h"
#include "kinova_driver/kinova_joint_trajectory_controller.h"

#include <sim_grasp/sim_graspModel.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <string>
using namespace std;
using namespace Eigen;

// #define PI 3.1415926

/*
该程序流程
1 在py程序中订阅抓取检测话题,运行神经网络,发布抓取检测结果   (py程序)
2 订阅抓取检测结果  (订阅器,回调函数)
    x:像素坐标
    y:像素坐标
    z:深度值 m
    angle:抓取角 弧度
    width:抓取宽度 m
3 将抓取位姿转换到机械臂base坐标系下 (回调函数)
4 执行抓取 (主程序)

机械手:0-张开 6400-闭合
*/


Eigen::Matrix3d camera_K(3, 3); // 相机内参
vector<double> arm_init_pose = {0, -0.3, 0.272, M_PI, 0, M_PI/2};    // 机械臂初始化位姿 XYZ RPY,此时相机距离桌面0.6m
// vector<double> camera_end_tran = {0.015, 0.128, 0.155+0.09};             // XYZ 深度相机到机械臂末端的平移
// vector<double> camera_end_tran = {-0.10, 0, 0.155};  //(本人修改)
// vector<double> camera_end_quat = { 0.707, 0.000, 0.000,  0.707};    // Wxyz 相机到机械臂末端的四元数

vector<double> camera_end_tran = {0.033, 0.097, 0.133};  
vector<double> camera_end_quat = {-0.017, -0.02, 0.999,  0.033};    // Wxyz 相机到机械臂末端的四元数

vector<double> drop_object_pose = {-0.4, -0.3, 0.418, M_PI, 0, M_PI/2};    // 将物体放置于容器时的位姿

Eigen::Matrix4d camera_end_mat; // 相机 -> 抓取器  变换矩阵
Eigen::Matrix4d base_end_mat;   // base -> 抓取器 变换矩阵
Eigen::MatrixXd base_location(4, 1);     // 抓取点在base坐标系的坐标
double angle_z = 0; // 抓取时,抓取器沿z轴的旋转角
double grasp_depth = 0; // 抓取深度,实际抓取点距离物体表面的深度
float grasp_width = 0.0f;
bool isgrasp = false;   // 抓取标志位,获取抓取检测结果后置true,抓取完成后置false
float gripper_openSize = 6400.0f;


void update_param(kinova::KinovaArm &kinova_arm)
{
    /*
    获得三个参数: 相机内参  相机->抓取器的变换矩阵  base->抓取器的变换矩阵
    相机内参和相机->抓取器的变换矩阵是定值,不需要更新
    base->抓取器的变换矩阵 根据每次机械臂运动的位置不同需要更新
    */

    // 1 深度相机内参
    // camera_K << 616.9149780273438, 0.0, 312.5096130371094, 0.0, 616.8095703125, 248.8218994140625, 0.0, 0.0, 1.0;   // RGB相机的内参
    //! 改这里
    //! 内参 从realsense /camera/depth/camera_info 话题读取(已修改)
    camera_K << 384.9385681152344, 0.0, 316.30389404296875, 0.0, 384.9385681152344, 239.39447021484375, 0.0, 0.0, 1.0;   // 深度相机的内参
    //camera_K << 385.25531005859375, 0.0, 320.9977111816406, 0.0, 385.25531005859375, 238.800537109375, 0.0, 0.0, 1.0;   // 深度相机的内参

    // 2 计算 相机 -> 抓取器  变换矩阵
    Eigen::Quaterniond camera_end_Q(camera_end_quat[0], camera_end_quat[1], camera_end_quat[2], camera_end_quat[3]);  // 相机 -> 抓取器的四元数
    Eigen::Matrix3d camera_end_rMat;
    camera_end_rMat = camera_end_Q.toRotationMatrix();    // 相机 -> 抓取器的旋转矩阵
    camera_end_mat << camera_end_rMat(0,0), camera_end_rMat(0,1) , camera_end_rMat(0,2), camera_end_tran[0],
                      camera_end_rMat(1,0), camera_end_rMat(1,1),  camera_end_rMat(1,2), camera_end_tran[1],
                      camera_end_rMat(2,0), camera_end_rMat(2,1),  camera_end_rMat(2,2), camera_end_tran[2],
                      0.0, 0.0, 0.0, 1.0;   

    // 3 计算 base -> 抓取器 变换矩阵
    // 读取机械臂当前位姿
    kinova::KinovaPose kinova_pose;
    kinova_arm.getCartesianPos(kinova_pose);

    Eigen::Vector3d base_end_euler(kinova_pose.ThetaX, kinova_pose.ThetaY, kinova_pose.ThetaZ);  // 欧拉角 xyz rpy
    Eigen::AngleAxisd rollAngle(AngleAxisd(base_end_euler(0), Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(AngleAxisd(base_end_euler(1), Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(AngleAxisd(base_end_euler(2), Vector3d::UnitZ()));
    
    Eigen::Matrix3d base_end_rMat;  // 旋转矩阵
    base_end_rMat = yawAngle * pitchAngle * rollAngle;

    base_end_mat << base_end_rMat(0,0), base_end_rMat(0,1) , base_end_rMat(0,2), kinova_pose.X,
                    base_end_rMat(1,0), base_end_rMat(1,1),  base_end_rMat(1,2), kinova_pose.Y,
                    base_end_rMat(2,0), base_end_rMat(2,1),  base_end_rMat(2,2), kinova_pose.Z,
                    0.0, 0.0, 0.0, 1.0;
    
    // 4 初始化 抓取点在抓取器坐标系中的位置
    base_location =  Eigen::MatrixXd::Identity(4, 1);   // 用单位矩阵初始化
}


void GraspResultCallback(const sim_grasp::sim_graspModel::ConstPtr& msg)
{
    // cout << "msg.x = " << msg->x << endl;
    // cout << "msg.y = " << msg->y << endl;
    // cout << "msg.z = " << msg->z << endl;
    // cout << "msg.grasp_depth = " << msg->grasp_depth << endl;
    // cout << "msg.angle = " << msg->angle << endl;
    // cout << "msg.width = " << msg->width << endl;

    if (msg->x == 1001)
        return;

    grasp_width = msg->width;

    // 将抓取位姿转换到机械臂base坐标系下

    // 1 *************** 像素坐标系 -> 相机坐标系 ***************
    Eigen::MatrixXd pixel_location(3, 1);   // 抓取点在像素坐标系
    pixel_location << msg->x * 1.0, msg->y * 1.0, 1.0;

    Eigen::MatrixXd camera_location(3, 1);  // 抓取点在相机坐标系
    camera_location =  Eigen::MatrixXd::Identity(3,1);   //用单位矩阵初始化
    camera_location = camera_K.inverse() * pixel_location;

    camera_location(0, 0) = camera_location(0, 0) * msg->z;
    camera_location(1, 0) = camera_location(1, 0) * msg->z;
    camera_location(2, 0) = msg->z;


    // 2 *************** 相机坐标系 -> 抓取器坐标系 ***************
    Eigen::MatrixXd camera_location4(4, 1);
    camera_location4 << camera_location(0, 0), camera_location(1, 0), camera_location(2, 0), 1.0;

    Eigen::MatrixXd end_location(4, 1);     // 抓取点在抓取器坐标系
    end_location =  Eigen::MatrixXd::Identity(4, 1);   // 用单位矩阵初始化
    end_location = camera_end_mat.inverse() * camera_location4;
    // printf("end_location (x, y, z) = (%.3f, %.3f, %.3f) \n", end_location(0, 0), end_location(1, 0), end_location(2, 0));

    // 3 *************** 抓取器坐标系 -> 机械臂base坐标系 ***************
    base_location = base_end_mat * end_location;
    // printf("base_location (x, y, z) = (%.3f, %.3f, %.3f) \n", base_location(0, 0), base_location(1, 0), base_location(2, 0));

    // 4 计算抓取角
    angle_z = fmod(5*M_PI - msg->angle + M_PI/2, M_PI);

    grasp_depth = msg->grasp_depth;

    isgrasp = true;
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "sim_arm_grasp");
    ros::NodeHandle nh("~");
    boost::recursive_mutex api_mutex;

    bool is_first_init = true;
    string kinova_robotType = "j2s6s200";
    string kinova_robotName = "j2s6s200";

    // init kinova
    kinova::KinovaComm comm(nh, api_mutex, is_first_init,kinova_robotType);
    kinova::KinovaArm kinova_arm(comm, nh, kinova_robotType, kinova_robotName);
    kinova::KinovaPoseActionServer pose_server(comm, nh, kinova_robotType, kinova_robotName);
    kinova::KinovaAnglesActionServer angles_server(comm, nh);
    kinova::KinovaFingersActionServer fingers_server(comm, nh);
    kinova::JointTrajectoryController joint_trajectory_controller(comm, nh);    

    // 初始化订阅器
    ros::Subscriber grasp_posture_sub = nh.subscribe("/grasp/grasp_result", 1000, GraspResultCallback);
    // 控制机械臂的发布器
    ros::Publisher gripper_cmd_pub = nh.advertise<std_msgs::String>("/grasp/gripper_cmd", 1000);
    // 初始化发布话题
    std_msgs::String gripper_cmd_msg;

    ros::Duration(2.0).sleep();

    string gripper_cmd = "a";
    gripper_cmd_msg.data = gripper_cmd;
    gripper_cmd_pub.publish(gripper_cmd_msg);
    ros::Duration(2.0).sleep();
    gripper_cmd = "r";
    gripper_cmd_pub.publish(gripper_cmd_msg);

    update_param(kinova_arm); // 更新 相机参数和坐标系关系矩阵


    /**************** 主程序 ******************/

    // 设置watch位姿
    ROS_INFO("move to watch");
    kinova_msgs::AddPoseToCartesianTrajectory::Response res;
    kinova_msgs::AddPoseToCartesianTrajectory::Request req_watch;
    req_watch.X = arm_init_pose[0];
    req_watch.Y = arm_init_pose[1];
    req_watch.Z = arm_init_pose[2];
    req_watch.ThetaX = arm_init_pose[3];
    req_watch.ThetaY = arm_init_pose[4];
    req_watch.ThetaZ = arm_init_pose[5];
    kinova_arm.addCartesianPoseToTrajectory(req_watch, res);  // 直接退出函数，不是等运动到目标位置再退出
    ros::Duration(3.0).sleep();

    kinova_arm.addCartesianPoseToTrajectory(req_watch, res);  // 直接退出函数，不是等运动到目标位置再退出
    ros::Duration(1.0).sleep();

    // 初始化抓取位姿
    kinova_msgs::AddPoseToCartesianTrajectory::Request req_grasp;
    req_grasp.ThetaX = arm_init_pose[3];
    req_grasp.ThetaY = arm_init_pose[4];

    // 初始化放置物体位姿
    kinova_msgs::AddPoseToCartesianTrajectory::Request req_drop;
    req_drop.X = drop_object_pose[0];
    req_drop.Y = drop_object_pose[1];
    req_drop.Z = drop_object_pose[2];
    req_drop.ThetaX = drop_object_pose[3];
    req_drop.ThetaY = drop_object_pose[4];
    req_drop.ThetaZ = drop_object_pose[5];

    ros::Rate loop_rate(10);
    
    int grasp_count = 0;

    ROS_INFO("start loop waiting ...");

    // 闭合机械手
    // gripper_cmd_msg.data = "220";
    // gripper_cmd_pub.publish(gripper_cmd_msg);

    while(ros::ok())
    {
        if (isgrasp)
        {
            ROS_INFO("***************** grasp count %d *****************", ++grasp_count);

            // 1 打开抓取器 抓取宽度->机械手指令
            //! 改这里
            //gripper_cmd = to_string(int(220. - grasp_width*1500. - 8));
            //gripper_cmd_msg.data = gripper_cmd;
            //gripper_cmd_pub.publish(gripper_cmd_msg);
            //gripper_openSize = -37647.06 * grasp_width + 6400.0-200;
            gripper_openSize = 3200.0f;
            if (gripper_openSize > 5000)
                gripper_openSize -= 200;
            kinova_arm.setFingerPos(gripper_openSize, gripper_openSize, 0);            
            ROS_INFO("open gripper");
            ros::Duration(1.0).sleep();
            
            // 2 运动到抓取位姿
            // 先移动到抓取点上方5cm处,再移动到抓取点,防止碰到物体
            req_grasp.X = base_location(0, 0);
            req_grasp.Y = base_location(1, 0);
            req_grasp.Z = base_location(2, 0) + 0.03;
            req_grasp.ThetaZ = angle_z;
            kinova_arm.addCartesianPoseToTrajectory(req_grasp, res);
            ros::Duration(2).sleep();

            req_grasp.Z = base_location(2, 0) - grasp_depth;
            kinova_arm.addCartesianPoseToTrajectory(req_grasp, res);
            ROS_INFO("move to grasp");

            // 延时等待运动至设定位置,比对抓取器位置
            int n = 10;
            while( abs(kinova_arm.getCartesianZ() - req_grasp.Z) > 0.01 and n-- > 0)
                ros::Duration(0.5).sleep();     // 通过while比对 替换 延时
            ros::Duration(0.5).sleep(); 
                
            // 3 关闭抓取器
            //! 改这里
            //gripper_cmd = to_string(255);
            //gripper_cmd_msg.data = gripper_cmd;
            //gripper_cmd_pub.publish(gripper_cmd_msg);
            gripper_openSize = 6400.0f;  // 通过msg->width获得
            kinova_arm.setFingerPos(gripper_openSize, gripper_openSize, 0);
            ROS_INFO("close gripper");
            ros::Duration(1.0).sleep();

            // 4 回到watch位姿
            req_watch.Z += 0.2;
            kinova_arm.addCartesianPoseToTrajectory(req_watch, res);
            req_watch.Z -= 0.2;
            ROS_INFO("move to watch");
            // 延时等待运动至设定位置,比对抓取器位置
            n = 6;
            while( abs(kinova_arm.getCartesianZ() - req_watch.Z+0.2) > 0.01 and n-- > 0)
                ros::Duration(0.5).sleep();     // 通过while比对 替换 延时
            ros::Duration(0.5).sleep(); 

            // 5 将物体放置于容器
            kinova_arm.addCartesianPoseToTrajectory(req_drop, res);  // 直接退出函数，不是等运动到目标位置再退出
            ros::Duration(3.0).sleep();

            // 6 打开抓取器
            //! 改这里
            //gripper_cmd = to_string(0);
            //gripper_cmd_msg.data = gripper_cmd;
            //gripper_cmd_pub.publish(gripper_cmd_msg);
            gripper_openSize = 0.0f;  // 通过msg->width获得
            kinova_arm.setFingerPos(gripper_openSize, gripper_openSize, 0);
            ROS_INFO("open gripper");
            ros::Duration(1.0).sleep();

            // 7 回到watch位姿
            kinova_arm.addCartesianPoseToTrajectory(req_watch, res);
            ROS_INFO("move to watch");
            // 延时等待运动至设定位置,比对抓取器位置
            n = 10;
            while( abs(kinova_arm.getCartesianX() - req_watch.X) > 0.01 and n-- > 0) {
                ros::Duration(0.5).sleep();     // 通过while比对 替换 延时
            }

            // 更新参数和状态
            update_param(kinova_arm); // 更新 相机参数和坐标系关系矩阵
            isgrasp = false;
            ROS_INFO("update param");

            ROS_INFO("=======================================================================================================");
        }

        ros::spinOnce();    // 处理订阅的回调函数
        loop_rate.sleep();  // 按照设置的频率延时
    }

    ros::spin();

    return 0;
}

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
#include <vector>
#include <string>
#include <fstream>
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


. z轴坐标为0.02时,抓取器闭合时的末端碰到桌面
. 使用三指模式或二指模式时,修改camera_end_tran
    二指: camera_end_tran = {0.024, 0.085, 0.155}
. 控制手指位置 finger: 0-全开 6400-全闭
    kinova_arm.setFingerPos(finger, finger, finger);
*/



Eigen::Matrix3d camera_K(3, 3); // 相机内参
vector<double> arm_init_pose = {0, -0.3, 0.318, M_PI, 0, 0};    // 机械臂初始化位姿 XYZ RPY  默认 0, -0.2, 0.4, PI, 0, 0
vector<double> camera_end_tran = {0.0259, 0.0772, -0.1663};  //XYZ 深度相机到机械臂末端的平移（钊效禹 2025.1.9.下午16：36）
vector<double> camera_end_quat = {0.0481412964327, -0.00630015730501, 0.0106926708921, -0.998763430641};    // Wxyz 相机到机械臂末端的四元数 -（钊效禹 2025.1.9.下午16：36）
//vector<double> camera_end_tran = {0.053, 0.098, 0.190-0.04};  //XYZ 深度相机到机械臂末端的平移（本人修改）
//vector<double> camera_end_quat = {0.118, -0.012, -0.035, 0.992};    // Wxyz 相机到机械臂末端的四元数 -
//查看标定结果方法：rosrun tf tf_echo /camera_color_optical_frame /j2s6s300_end_effector; eg. 平移结果不变；原始姿态结果是[0.017, -0.02, 0.999,  0.033],程序里需要修改为[0.033, -0.017, -0.02, 0.999]
vector<double> drop_object_pose = {-0.5, -0.3, 0.318, M_PI, 0, M_PI/2};    // 将物体放置于容器时的位姿
//vector<double> drop_object_pose = {0.47, -0.3, 0.318, M_PI, 0, M_PI/2};    // 将物体放置于容器时的位姿

Eigen::Matrix4d camera_end_mat; // 相机 -> 抓取器  变换矩阵
Eigen::Matrix4d base_end_mat;   // base -> 抓取器 变换矩阵
Eigen::MatrixXd base_location(4, 1);     // 抓取点在base坐标系的坐标
Eigen::MatrixXd base_location_2(4, 1);
double angle_z = 0; // 抓取时,抓取器沿z轴的旋转角
bool isgrasp = false;   // 抓取标志位,获取抓取检测结果后置true,抓取完成后置false

float gripper_openSize = 6400.0f;   // 抓取器张开尺寸 0-全开 6400-全闭
float grasp_width = 0.0f;
int capture_flag = 0;
kinova_msgs::AddPoseToCartesianTrajectory::Request req_drop; // 初始化放置物体位姿的变量

void update_param(kinova::KinovaArm &kinova_arm)
{
    /*
    获得三个参数: 相机内参  相机->抓取器的变换矩阵  base->抓取器的变换矩阵
    相机内参和相机->抓取器的变换矩阵是定值,不需要更新
    base->抓取器的变换矩阵 根据每次机械臂运动的位置不同需要更新
    */

    // 1 深度相机内参
    camera_K << 614.6087646484375, 0.0, 322.89813232421875, 0.0, 614.5292358398438, 237.93869018554688, 0.0, 0.0, 1.0;    
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
    base_location_2 =  Eigen::MatrixXd::Identity(4, 1);
}

// void imageCallback(const sensor_msgs::ImageConstPtr& msg) // 回调函数，用于接收"/camera/color/image_raw"话题的消息
// {
//     // 将msg中的数据存入数组
//     if (capture_flag ==1)
//     {        
//         image_msg = *msg;        
//     }          
               
// }

void GraspResultCallback(const sim_grasp::sim_graspModel::ConstPtr& msg)
{
    // cout << "msg.x = " << msg->x << endl;
    // cout << "msg.y = " << msg->y << endl;
    // cout << "msg.z = " << msg->z << endl;
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
     printf("base_location (x, y, z) = (%.3f, %.3f, %.3f) \n", base_location(0, 0), base_location(1, 0), base_location(2, 0));

    // 4 计算抓取角
    angle_z = fmod(5.0*M_PI-msg->angle, 2.0*M_PI);    

   // ***********************计算放置物品的坐标点*************************************************************************************
    float drop_x, drop_y, drop_z; 
    //指定文件路径
    std::string file_path = "/home/guoxy/Camera_capture/box_destination.txt";    
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        ROS_WARN("Error: File 'box.txt' not found in the current directory.");        
    }
    //读取box.txt文件中的float类型数据并存入drop_data数组
    std::vector<float> drop_data;
    float value;
    while (file >> value)
    {
        drop_data.push_back(value);
    }
    file.close();
    drop_x=int((drop_data[0]+drop_data[2]))/2;
    drop_y=int((drop_data[1]+drop_data[3]))/2;
    // drop_x=372;
    // drop_y=282;
    drop_z=msg->z;
    
    //将放置位姿转换到机械臂base坐标系下
    //1 *************** 像素坐标系 -> 相机坐标系 *************** 
    Eigen::MatrixXd pixel_location_II(3, 1);   // 抓取点在像素坐标系   
    pixel_location_II << drop_x * 1.0, drop_y * 1.0, 1.0;
    
    Eigen::MatrixXd camera_location_II(3, 1);  // 抓取点在相机坐标系
    camera_location_II =  Eigen::MatrixXd::Identity(3,1);   //用单位矩阵初始化
    camera_location_II = camera_K.inverse() * pixel_location_II;

    camera_location_II(0, 0) = camera_location_II(0, 0) * drop_z;
    camera_location_II(1, 0) = camera_location_II(1, 0) * drop_z;
    camera_location_II(2, 0) = drop_z;

    // 2 *************** 相机坐标系 -> 抓取器坐标系 ***************
    Eigen::MatrixXd camera_location4_II(4, 1);    
    camera_location4_II << camera_location_II(0, 0), camera_location_II(1, 0), camera_location_II(2, 0), 1.0;
    
    Eigen::MatrixXd end_location_II(4, 1);     // 抓取点在抓取器坐标系
    end_location_II =  Eigen::MatrixXd::Identity(4, 1);   // 用单位矩阵初始化
    end_location_II = camera_end_mat.inverse() * camera_location4_II;
    // printf("end_location (x, y, z) = (%.3f, %.3f, %.3f) \n", end_location(0, 0), end_location(1, 0), end_location(2, 0));

    // 3 *************** 抓取器坐标系 -> 机械臂base坐标系 ***************
    base_location_2 = base_end_mat * end_location_II; 
     printf("base_location_2 (x, y, z) = (%.3f, %.3f, %.3f) \n", base_location_2(0, 0), base_location_2(1, 0), base_location_2(2, 0));       
    //***************************************************************************************************************************

    isgrasp = true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sim_arm_grasp");
    ros::NodeHandle nh("~");
    boost::recursive_mutex api_mutex;

    bool is_first_init = true;
    std::string kinova_robotType = "";
    std::string kinova_robotName = "";

    kinova_robotType = argv[argc-1];
    kinova_robotName = kinova_robotType;
    ROS_INFO("kinova_robotType is %s.", kinova_robotType.c_str());

    // init kinova
    kinova::KinovaComm comm(nh, api_mutex, is_first_init,kinova_robotType);
    kinova::KinovaArm kinova_arm(comm, nh, kinova_robotType, kinova_robotName);
    kinova::KinovaPoseActionServer pose_server(comm, nh, kinova_robotType, kinova_robotName);
    kinova::KinovaAnglesActionServer angles_server(comm, nh);
    kinova::KinovaFingersActionServer fingers_server(comm, nh);
    kinova::JointTrajectoryController joint_trajectory_controller(comm, nh);    

    // 初始化订阅器
    ros::Subscriber grasp_posture_sub = nh.subscribe("/grasp/grasp_result", 1000, GraspResultCallback);
    //ros::Subscriber capture_again_sub = nh.subscribe("/Capture_picture", 100, imageCallback); 
    // 初始化发布器    
    ros::Publisher capture_again = nh.advertise<std_msgs::String>("/Capture_picture", 10);

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

    // 初始化抓取位姿
    kinova_msgs::AddPoseToCartesianTrajectory::Request req_grasp;
    req_grasp.ThetaX = arm_init_pose[3];
    req_grasp.ThetaY = arm_init_pose[4];

    // 初始化放置物体位姿
    //kinova_msgs::AddPoseToCartesianTrajectory::Request req_drop;
    
    req_drop.X = drop_object_pose[0];
    req_drop.Y = drop_object_pose[1];
    req_drop.Z = drop_object_pose[2];  
    req_drop.ThetaX = drop_object_pose[3];
    req_drop.ThetaY = drop_object_pose[4];
    req_drop.ThetaZ = drop_object_pose[5];


    ros::Rate loop_rate(10);
    
    int grasp_count = 0;    
    while(ros::ok())
    {
        if (isgrasp)
        {
            ROS_INFO("***************** grasp count %d *****************", ++grasp_count);
            

            // 1 打开抓取器 米->手指位置
            //! kinova gripper control, 参考这里的代码
           // gripper_openSize = -37647.06 * grasp_width + 6400.0-200;
            //if (gripper_openSize > 5000)
                //gripper_openSize -= 200;            
            gripper_openSize = 6400/2;
            kinova_arm.setFingerPos(gripper_openSize, gripper_openSize, 0);
            ROS_INFO("1 open gripper");
            
            // 2 运动到抓取位姿
            // 先移动到抓取点上方5cm处,再移动到抓取点,防止碰到物体
            req_grasp.X = base_location(0, 0); 
            req_grasp.Y = base_location(1, 0);
            req_grasp.Z = max(base_location(2, 0) - 0.03, 0.02) + 0.05;
            req_grasp.ThetaZ = angle_z; // + 0.175;
            kinova_arm.addCartesianPoseToTrajectory(req_grasp, res);

            req_grasp.Z = max(base_location(2, 0) - 0.03, 0.02);
            kinova_arm.addCartesianPoseToTrajectory(req_grasp, res);
            ROS_INFO("2 move to grasp");

            //延时等待运动至设定位置,比对抓取器位置
            int n = 20;
            //while( abs(kinova_arm.getCartesianZ() - req_grasp.Z) > 0.005 and n-- > 0)
            while( abs(kinova_arm.getCartesianZ() - req_grasp.Z) > 0.005 and n-- > 0)  
                ros::Duration(0.5).sleep();     // 通过while比对 替换 延时
            ros::Duration(0.5).sleep();     // 通过while比对 替换 延时
                
            // 3 关闭抓取器
            gripper_openSize = 6400.0f;  // 通过msg->width获得
            kinova_arm.setFingerPos(gripper_openSize, gripper_openSize, 0);
            ros::Duration(1).sleep();     // 通过while比对 替换 延时
            ROS_INFO("3 close gripper");

            // 4 回到watch位姿
            req_watch.X = req_grasp.X;
            req_watch.Y = req_grasp.Y;
            req_watch.Z = arm_init_pose[2]/2;
            //req_watch.ThetaZ = req_grasp.ThetaZ; //req_watch.ThetaZ改动会导致相片产生旋转！
            kinova_arm.addCartesianPoseToTrajectory(req_watch, res);            
            // 延时等待运动至设定位置,比对抓取器位置
            n = 20;
            //while( abs(kinova_arm.getCartesianZ() - req_watch.Z) > 0.01 and n-- > 0)
            while( abs(kinova_arm.getCartesianZ() - req_watch.Z) > 0.01 )
            {
                ros::Duration(0.5).sleep();     // 通过while比对 替换 延时
            }
            ROS_INFO("4 move to watch");
            ros::Duration(0.5).sleep();

            // 5 将物体放置于容器                       
            req_drop.X = base_location_2(0, 0);  //放置物品位置的x坐标
            req_drop.Y = base_location_2(1, 0);  //放置物品位置的y坐标
            req_drop.Z = max(base_location_2(2, 0) - 0.03, 0.02);; //放置物品位置的z坐标
            kinova_arm.addCartesianPoseToTrajectory(req_drop, res);  // 直接退出函数，不是等运动到目标位置再退出
            ros::Duration(3.0).sleep();

            // 6 打开抓取器
            gripper_openSize = 0.0f;  // 通过msg->width获得
            kinova_arm.setFingerPos(gripper_openSize, gripper_openSize, 0);
            ros::Duration(1.0).sleep();     // 通过while比对 替换 延时
            ROS_INFO("6 open gripper");

            // 7 回到watch位姿
            req_watch.X = arm_init_pose[0];
            req_watch.Y = arm_init_pose[1];
            req_watch.Z = arm_init_pose[2];            
            //req_watch.ThetaZ = arm_init_pose[5];
            kinova_arm.addCartesianPoseToTrajectory(req_watch, res);            
            // 延时等待运动至设定位置,比对抓取器位置
            n = 30;
            //while( abs(kinova_arm.getCartesianX() - req_watch.X) > 0.01 and n-- > 0)
            while( abs(kinova_arm.getCartesianZ() - req_watch.Z) > 0.01 )
            {
                ros::Duration(0.5).sleep();     // 通过while比对 替换 延时
            }
            ROS_INFO("7 move to watch");                      
            ros::Duration(0.5).sleep();                         

            // 更新参数和状态            
            update_param(kinova_arm); // 更新 相机参数和坐标系关系矩阵
            ROS_INFO("8 update param");
            isgrasp = false;                        
            capture_flag = 1;       
        }        
        ros::spinOnce();    // 处理订阅的回调函数               
        // if (capture_flag == 1) 
        // {
        //     std_msgs::String capture_msg;
        //     capture_msg.data = "OK!！";        // 发布再次拍照命令
        //     capture_again.publish(capture_msg);                            
        //     capture_flag = 0;            
        // }
        loop_rate.sleep();  // 按照设置的频率延时
    }

    ros::spin();

    return 0;
}

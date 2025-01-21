//============================================================================
// Name        : kinova_arm_driver.cpp
// Author      : WPI, Clearpath Robotics
// Version     : 0.5
// Copyright   : BSD
// Description : A ROS driver for controlling the Kinova Kinova robotic manipulator arm
//============================================================================

#include "kinova_driver/kinova_api.h"
#include "kinova_driver/kinova_arm.h"
#include "kinova_driver/kinova_tool_pose_action.h"
#include "kinova_driver/kinova_joint_angles_action.h"
#include "kinova_driver/kinova_fingers_action.h"
#include "kinova_driver/kinova_joint_trajectory_controller.h"
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "kinova_arm_driver");
    ros::NodeHandle nh("~");
    boost::recursive_mutex api_mutex;

    bool is_first_init = true;
    std::string kinova_robotType = "";
    std::string kinova_robotName = "";

    // Retrieve the (non-option) argument:
    if ( (argc <= 1) || (argv[argc-1] == NULL) ) // there is NO input...
    {
        std::cerr << "No kinova_robotType provided in the argument!" << std::endl;
        return -1;
    }
    else // there is an input...
    {
        kinova_robotType = argv[argc-1];
        ROS_INFO("kinova_robotType is %s.", kinova_robotType.c_str());
        if (!nh.getParam("robot_name", kinova_robotName))
        {
          kinova_robotName = kinova_robotType;
        }
        ROS_INFO("kinova_robotName is %s.", kinova_robotName.c_str());
    }


    while (ros::ok())
    {
        try
        {
            kinova::KinovaComm comm(nh, api_mutex, is_first_init,kinova_robotType);
            kinova::KinovaArm kinova_arm(comm, nh, kinova_robotType, kinova_robotName);
            kinova::KinovaPoseActionServer pose_server(comm, nh, kinova_robotType, kinova_robotName);
            kinova::KinovaAnglesActionServer angles_server(comm, nh);
            kinova::KinovaFingersActionServer fingers_server(comm, nh);
            kinova::JointTrajectoryController joint_trajectory_controller(comm, nh);

            /*以下为测试代码*/
            ros::Duration(3.0).sleep();

            cout << endl;
            cout << "************************************" << endl;
            cout << "move to test1" << endl;
            kinova_msgs::AddPoseToCartesianTrajectory::Response res;
            kinova_msgs::AddPoseToCartesianTrajectory::Request req;
            req.X = 0.0330119;
            req.Y = -0.453956;
            req.Z = 0.326703;
            req.ThetaX = 3.1415926;
            req.ThetaY = 0;
            req.ThetaZ = 0;
            kinova_arm.addCartesianPoseToTrajectory(req, res);  // 直接退出函数，不是等运动到目标位置再退出
            ros::Duration(5.0).sleep();

            int m = 10;
            while(m--)
            {
                int n = 10;
                while(n--)
                {
                    cout << "test1-" << n << endl;
                    req.Y -= 0.01;
                    kinova_arm.addCartesianPoseToTrajectory(req, res);
                    ros::Duration(1.0).sleep();
                }
                cout << "move to test1" << endl;
                req.Y = -0.453956;
                kinova_arm.addCartesianPoseToTrajectory(req, res);  // 直接退出函数，不是等运动到目标位置再退出
                ros::Duration(5.0).sleep();
            }

            ros::spin();
        }
        catch(const std::exception& e)
        {
            ROS_ERROR_STREAM(e.what());
            kinova::KinovaAPI api;
            boost::recursive_mutex::scoped_lock lock(api_mutex);
            api.closeAPI();
            ros::Duration(1.0).sleep();
        }

        is_first_init = false;
    }
    return 0;
}

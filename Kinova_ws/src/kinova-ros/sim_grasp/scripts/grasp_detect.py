#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import time
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from std_msgs.msg import String
from sim_grasp.msg import sim_graspModel
import base64
import numpy as np
import torch
from skimage.draw import line
from sgdn import SGDN

import sys
try:
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
except:
    pass
from cv_bridge import CvBridge, CvBridgeError
import cv2



def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img

def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

# !!!! 改计算方法
# def length2pixel(length, depth_val, img_h=480, img_w=848):
#     """
#     将单位为m的长度值转换为像素长度
#     length: float 单位为m
#     depth_val: float 深度值(抓取点的深度值)

#     return: 单位为像素的长度值,float
#     """
#     # 相机内参
#     camera_K = np.array([
#         [418.4587707519531, 0.0, 422.0575866699219], 
#         [0.0, 418.4587707519531, 241.18592834472656],
#         [0.0, 0.0, 1.0]
#     ])

#     # 计算虚拟点在相机坐标系的坐标
#     Xc = length / (2 * (2**0.5))
#     Yc = Xc
#     loc_camera = np.array([
#         [Xc],
#         [Yc],
#         [depth_val]
#     ])

#     # 根据内参矩阵计算该虚拟点在像素坐标系的坐标
#     loc_pixel = np.matmul(camera_K, loc_camera) / depth_val

#     # 根据loc_pixel的[u,v]计算最终的像素长度
#     return (((loc_pixel[0][0] - (img_w/2)) ** 2 + (loc_pixel[1][0] - (img_h/2)) ** 2) ** 0.5) * 2

def drawGrasps(img, grasps, im_dep):
    """
    绘制grasp
    file: img路径
    grasps: list()	元素是 [row, col, angle, width, conf]
    width: 米
    """
    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width, conf = grasp
        row = int(row)
        col = int(col)
        if im_dep is not None:
            depth = im_dep[row, col]
            # 方法1:使用固定值
            # width_pixel = width * 971.66  # 米->像素  888.89     固定深度
            # 方法2:根据抓取点的深度
            # width_pixel = length2pixel(width, depth) / 2
            width_pixel = length_TO_pixels(width, depth) / 2
        else:
            width_pixel = width

        angle2 = calcAngle2(angle)
        k = math.tan(angle)

        if k == 0:
            dx = width_pixel
            dy = 0
        else:
            dx = k / abs(k) * width_pixel / pow(k ** 2 + 1, 0.5)
            dy = k * dx

        if angle < math.pi:
            cv2.arrowedLine(img, (col, row), (int(col + dx), int(row - dy)), (255, 255, 255), 2, 8, 0, 0.3)
        else:
            cv2.arrowedLine(img, (col, row), (int(col - dx), int(row + dy)), (255, 255, 255), 2, 8, 0, 0.3)

        if angle2 < math.pi:
            cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 255, 255), 2)
        else:
            cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 255, 255), 2)

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255
        
        # img[row, col] = [color_b, color_g, color_r]
        cv2.circle(img, (col, row), 3, (color_b, color_g, color_r), -1)

    return img


def minDepth(img, row, col, l):
    """
    获取(row, col)周围l*l范围内最小的深度值
    :param img: 单通道深度图
    :param l: 3, 5, 7, 9 ...
    :return: float
    """
    row_t = row - (l - 1) / 2
    row_b = row + (l - 1) / 2
    col_l = col - (l - 1) / 2
    col_r = col + (l - 1) / 2

    min_depth = 10000

    for r in range(row_b + 1)[row_t + 1:]:
        for c in range(col_r + 1)[col_l + 1:]:
            dep = img[int(r), int(c)]
            if 1 < dep < min_depth:
                min_depth = dep
            # print('row = {}, col = {}'.format(r, c))
    return min_depth

def ptsOnRect(pts):
    """
    获取矩形框上五条线上的点
    五条线分别是：四条边缘线，1条对角线
    pts: np.array, shape=(4, 2) (row, col)
    """
    rows1, cols1 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[1, 0]), int(pts[1, 1]))
    rows2, cols2 = line(int(pts[1, 0]), int(pts[1, 1]), int(pts[2, 0]), int(pts[2, 1]))
    rows3, cols3 = line(int(pts[2, 0]), int(pts[2, 1]), int(pts[3, 0]), int(pts[3, 1]))
    rows4, cols4 = line(int(pts[3, 0]), int(pts[3, 1]), int(pts[0, 0]), int(pts[0, 1]))
    rows5, cols5 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[2, 0]), int(pts[2, 1]))

    rows = np.concatenate((rows1, rows2, rows3, rows4, rows5), axis=0)
    cols = np.concatenate((cols1, cols2, cols3, cols4, cols5), axis=0)
    return rows, cols

def ptsOnRotateRect(pt1, pt2, w, img_dep_rgb):
    """
    绘制矩形
    已知图像中的两个点（x1, y1）和（x2, y2），以这两个点为端点画线段，线段的宽是w。这样就在图像中画了一个矩形。
    pt1: [row, col] 
    w: 单位像素
    img: 绘制矩形的图像, 单通道
    """
    y1, x1 = pt1
    y2, x2 = pt2

    if x2 == x1:
        if y1 > y2:
            angle = math.pi / 2
        else:
            angle = 3 * math.pi / 2
    else:
        tan = (y1 - y2) * 1.0 / (x2 - x1)
        angle = np.arctan(tan)

    points = []
    points.append([y1 - w / 2 * np.cos(angle), x1 - w / 2 * np.sin(angle)])
    points.append([y2 - w / 2 * np.cos(angle), x2 - w / 2 * np.sin(angle)])
    points.append([y2 + w / 2 * np.cos(angle), x2 + w / 2 * np.sin(angle)])
    points.append([y1 + w / 2 * np.cos(angle), x1 + w / 2 * np.sin(angle)])
    points = np.array(points)

    cv2.circle(img_dep_rgb, (int(points[0][1]), int(points[0][0])), 3, (0, 0, 0), -1) 
    cv2.circle(img_dep_rgb, (int(points[1][1]), int(points[1][0])), 3, (0, 0, 0), -1) 
    cv2.circle(img_dep_rgb, (int(points[2][1]), int(points[2][0])), 3, (0, 0, 0), -1) 
    cv2.circle(img_dep_rgb, (int(points[3][1]), int(points[3][0])), 3, (0, 0, 0), -1) 

    # 方案1，比较精确，但耗时
    # rows, cols = polygon(points[:, 0], points[:, 1], (10000, 10000))	# 得到矩形中所有点的行和列

    # 方案2，速度快
    return ptsOnRect(points)	# 得到矩形中所有点的行和列

def collision_detection(pt, dep, angle, depth_map, finger_l1, finger_l2, img_dep_rgb):
    """
    碰撞检测
    pt: (row, col)
    angle: 抓取角 弧度
    depth_map: 深度图
    finger_l1 l2: 像素长度

    return:
        True: 无碰撞
        False: 有碰撞
    """
    row, col = pt
    H, W = depth_map.shape

    # 两个点
    row1 = int(row - finger_l2 * math.sin(angle))
    col1 = int(col + finger_l2 * math.cos(angle))

    row1 = max(min(row1, H-1), 0)
    col1 = max(min(col1, W-1), 0)

    cv2.circle(img_dep_rgb, (col1, row1), 3, (0, 0, 0), -1) 
    
    # 在截面图上绘制抓取器矩形
    # 检测截面图的矩形区域内是否有1
    rows, cols = ptsOnRotateRect([row, col], [row1, col1], finger_l1, img_dep_rgb)

    try:
        # print('np.min(depth_map[rows, cols]) = ', np.min(depth_map[rows, cols]))
        # print('dep = ', dep)
        if np.min(depth_map[rows, cols]) > dep:   # 无碰撞
            # print('无碰撞')
            return True
    except:
        return True    # 有碰撞
    
    return False    # 有碰撞

def getGraspDepth(camera_depth, grasp_row, grasp_col, grasp_angle, grasp_width, finger_l1, finger_l2, img_dep_rgb):
    """
    根据深度图像及抓取角、抓取宽度，计算最大的无碰撞抓取深度（相对于物体表面的下降深度）
    此时抓取点为深度图像的中心点
    camera_depth: 位于抓取点正上方的相机深度图
    grasp_angle：抓取角 弧度
    grasp_width：抓取宽度 像素
    finger_l1 l2: 抓取器尺寸 像素长度

    return: 抓取深度，相对于相机的深度
    """
    # grasp_row = int(camera_depth.shape[0] / 2)
    # grasp_col = int(camera_depth.shape[1] / 2)
    # 首先计算抓取器两夹爪的端点
    k = math.tan(grasp_angle)
    H, W = camera_depth.shape

    grasp_width /= 2
    if k == 0:
        dx = grasp_width
        dy = 0
    else:
        dx = k / abs(k) * grasp_width / pow(k ** 2 + 1, 0.5)
        dy = k * dx
    
    pt1 = (max(min(int(grasp_row - dy), H-1), 0), max(min(int(grasp_col + dx), W-1), 0))
    pt2 = (max(min(int(grasp_row + dy), H-1), 0), max(min(int(grasp_col - dx), W-1), 0))

    # print('grasp_row, grasp_col = ', grasp_row, grasp_col)
    # print('pt1 = ', pt1[0], pt1[1])
    # print('pt2 = ', pt2[0], pt2[1])

    cv2.circle(img_dep_rgb, (pt1[1], pt1[0]), 3, (0, 0, 0), -1) 
    cv2.circle(img_dep_rgb, (pt2[1], pt2[0]), 3, (0, 0, 0), -1) 

    # 下面改成，从抓取线上的最高点开始向下计算抓取深度，直到碰撞或达到最大深度
    rr, cc = line(pt1[0], pt1[1], pt2[0], pt2[1])   # 获取抓取线路上的点坐标
    min_depth = np.min(camera_depth[rr, cc])
    # print('camera_depth[grasp_row, grasp_col] = ', camera_depth[grasp_row, grasp_col])
    # print('min_depth = ', min_depth)

    grasp_depth = min_depth + 0.003
    while grasp_depth < min_depth + 0.05:
        # print('--1--')
        if not collision_detection(pt1, grasp_depth, grasp_angle, camera_depth, finger_l1, finger_l2, img_dep_rgb):
            return max(grasp_depth - 0.003 - min_depth, 0.03)
        # print('--2--')
        if not collision_detection(pt2, grasp_depth, grasp_angle + math.pi, camera_depth, finger_l1, finger_l2, img_dep_rgb):
            return max(grasp_depth - 0.003 - min_depth, 0.03)
        grasp_depth += 0.003

    return grasp_depth - min_depth

def iscapturecallback(data):
    global rgb2_flag           
    rgb2_flag = 1 

def rgbCallback(data):
    global graspStateRGB       
    if graspStateRGB :
        img_rgb = bridge.imgmsg_to_cv2(data, "bgr8")  # (480, 640, 3)
        cv2.imwrite('/path/to/your/Camera_capture/rgb_before.png', img_rgb)
        ims_rgb.append(img_rgb)
        if len(ims_rgb) >= 10:
            ims_rgb.pop(0)
        #time.sleep(4) #拍照刷新
        graspStateRGB = False        

def rgb2Callback(data): 
    global rgb2_flag 
    if rgb2_flag ==1:     
        img_rgb_2 = bridge.imgmsg_to_cv2(data, "bgr8")  # (480, 640, 3)
        cv2.imwrite('/path/to/your/Camera_capture/rgb.png', img_rgb_2)                          
        rgb2_flag = 0

def depCallback(data):
    global graspStateD
    if graspStateD:
        img_dep = bridge.imgmsg_to_cv2(data, "16UC1")  # (480, 640)
        cv2.imwrite('/path/to/your/Camera_capture/depth.png', img_dep)
        ims_dep.append(img_dep)
        if len(ims_dep) >= 10:
            ims_dep.pop(0)
        graspStateD = False


def filterGrasp(grasps, im_dep, thresh=0.01):
    """
    筛选抓取配置: 抓取点的深度比两侧高
    grasps: list([row, col, angle, width, conf])
    angle: 弧度
    width: 米
    im_dep: np.float  单位m
    """
    H, W = im_dep.shape[:2]
    ret1 = []
    ret2 = []
    for grasp in grasps:
        row, col, angle, width, conf = grasp
        depth = im_dep[int(row), int(col)]

        # 方法1:使用固定值
        # width_pixel = width * 971.66  # 米->像素  888.89     固定深度
        # 方法2:根据抓取点的深度
        # width_pixel = length2pixel(width, depth)
        width_pixel = length_TO_pixels(width, depth)

        width_pixel = width_pixel / 2
        angle2 = calcAngle2(angle)
        k = math.tan(angle)

        if k == 0:
            dx = width_pixel
            dy = 0
        else:
            dx = k / abs(k) * width_pixel / pow(k ** 2 + 1, 0.5)
            dy = k * dx

        pt1 = [int(col + dx), int(row - dy)] if angle < math.pi else [int(col - dx), int(row + dy)]    # (x, y)
        pt1[0] = min(max(pt1[0], 0), W-1)
        pt1[1] = min(max(pt1[1], 0), H-1)
        pt2 = [int(col + dx), int(row - dy)] if angle2 < math.pi else [int(col - dx), int(row + dy)]    # (x, y)
        pt2[0] = min(max(pt2[0], 0), W-1)
        pt2[1] = min(max(pt2[1], 0), H-1)  

        # 计算两个端点的深度
        depth1 = im_dep[pt1[1], pt1[0]]
        depth2 = im_dep[pt2[1], pt2[0]]

        if (depth1 - depth >= thresh) and (depth2 - depth >= thresh) and row >= 30 and row < H-30 and col >= 30 and col < W-30:
            # return [grasp]
            ret1.append(grasp)
        elif row >= 30 and row < H-30 and col >= 30 and col < W-30:
            ret2.append(grasp)

    # return [grasps[0]]
    if len(ret1) > 0:
        print('return filter grasp')
        return ret1
    elif len(ret2) > 0:
        print('return origin grasp')
        return ret2
    else:
        return None

def depth2Gray(im_depth):
    """
    将深度图转至三通道8位灰度图
    (h, w, 3)
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    return ret

def depth2RGB(im_depth):
    """
    将深度图转至三通道8位彩色图
    先将值为0的点去除，然后转换为彩图，然后将值为0的点设为红色
    (h, w, 3)
    im_depth: 单位 mm
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color

def length_TO_pixels(l, dep):
    """
    与相机距离为dep的平面上 有条线，长l，获取这条线在图像中的像素长度
    l: m
    dep: m
    """
    return l * 385.25 / dep # 616.85为相机内参的 df/dx和df/dy的均值

def runGraspCallback(data):
    global graspStateRGB, graspStateD,rgb2_flag
    graspStateRGB = True
    graspStateD = True
    rgb2_flag = 1

    while graspStateD is True:
        time.sleep(0.1)

    # 获取用于抓取检测的图像
    # img_rgb = ims_rgb[-1]
    img_dep = ims_dep[-1]   # 单位mm
    img_dep = img_dep.astype(np.float) / 1000.0     # 单位m
    img_dep = inpaint(img_dep)  # 修复

    start_time = time.time()




    # 截取中间区域
    # with open('/path/to/your/Camera_capture/box.txt', 'r') as file:
    #     l, t, r, b = [int(float(line.strip())) for line in file]
    # size = 200  # 截取正方形的边长 (原始图像 H：480；W：640)
    # H, W = img_dep.shape
    # central_W=l+(r-l)/2
    # central_H=t+(b-t)/2
    # t=int(central_H-size/2)
    # l=int(central_W-size/2)
    # # #l = int((W - size) / 2)
    # # l = 0   
    # # t = int((H - size) / 2)          
    # r = l + size      
    # b = t + size   
    # img_dep_crop = img_dep[t:b, l:r] ###重要！视觉大模型接口！


    size = 480  # 434
    H, W = img_dep.shape
    l = int((W - size) / 2)
    t = int((H - size) / 2)
    r = l + size
    b = t + size
    img_dep_crop = img_dep[t:b, l:r]



    # zero_loc = np.where(img_dep_crop==0)
    # img_dep_crop = inpaint(img_dep_crop)  # 修复
    # img_rgb_crop = img_rgb[t:b, l:r, :]

    # print('np.unique(img_dep_crop) = ', np.unique(img_dep_crop))

    # 将 img_dep_crop 送入抓取检测程序，获取平面抓取参数
    # ================================================
    global sgdn
    # print('img_dep_crop.min(), img_dep_crop.max() = ', img_dep_crop.min(), img_dep_crop.max())
    grasps = sgdn.predict(img_dep_crop, device, mode='max', scale=0.4/0.7, thresh=0.1)
    # 测试用
    # grasps = [[200, 200, 0, 0.05, 1.0]]

    if len(grasps) == 0:
        col = 1001   # 未找到目标：1001  无法抓取：1002
        row = 0
        dep = 0
        grasp_depth = 0
        angle = 0
        width = 0
        conf = 0
        print('len(grasps) == 0')
    else:
        # 筛选抓取点
        grasps_filter = filterGrasp(grasps, img_dep_crop)
        if grasps_filter != None:
            grasps = grasps_filter

        row = int(grasps[0][0] + t)
        col = int(grasps[0][1] + l)
        dep = img_dep[row, col]
        # dep = minDepth(img_dep, int(row), int(col), 3) # 11
        angle = grasps[0][2]
        width = grasps[0][3]    # 米
        conf = grasps[0][4]
        # 计算 grasp_width, finger_l1, finger_l2
        finger_l1 = 0.02   # m
        finger_l2 = 0.005   # m
        # print('length_TO_pixels(finger_l1, dep) = ', length_TO_pixels(finger_l1, dep))
        # print('length_TO_pixels(finger_l2, dep) = ', length_TO_pixels(finger_l2, dep))
        img_dep_rgb = depth2RGB(img_dep)
        grasp_depth = getGraspDepth(img_dep, row, col, angle, 
                                    length_TO_pixels(width, dep), 
                                    length_TO_pixels(finger_l1, dep), 
                                    length_TO_pixels(finger_l2, dep), img_dep_rgb)

    sum_t = time.time() - start_time
    print('sum time: ', sum_t)
    dep_pub.publish(bridge.cv2_to_imgmsg(img_dep_rgb, "bgr8"))

    if col == 1001:
        grasp_pub.publish(1001, 0, 0.0, 0.0, 0.0)
        print('failed!')
    else:
        print('(y, x): ', row, col)
        print('depth: ', dep)
        print('grasp_depth: ', grasp_depth)
        print('angle: ', angle)
        print('width: ', width)
        print('confidence: ', conf)
        grasp_pub.publish(col, row, dep, grasp_depth, angle, width)        

        # 绘制预测结果
        img_dep_grasp = np.copy(img_dep_crop)
        img_dep_grasp = depth2RGB(img_dep_grasp)
        # img_dep_grasp[zero_loc] = (0, 0, 0)
        img_dep_grasp = drawGrasps(img_dep_grasp, grasps, img_dep_crop)
        # img_rgb_grasp = drawGrasps(img_rgb_crop, grasps, img_dep_crop)
        # 发布抓取检测结果图像
        grasp_img_dep_pub.publish(bridge.cv2_to_imgmsg(img_dep_grasp, "bgr8"))
        # grasp_img_rgb_pub.publish(bridge.cv2_to_imgmsg(img_rgb_grasp, "bgr8"))


if __name__ == '__main__':
    try:
        # ROS节点初始化
        rospy.init_node('grasp_detect', anonymous=False)

        # 发布器初始化
        grasp_pub = rospy.Publisher('/grasp/grasp_result', sim_graspModel, queue_size=1000)
        grasp_img_dep_pub = rospy.Publisher('/grasp/grasp_img_dep', Image, queue_size=1000)
        dep_pub = rospy.Publisher('/grasp/dep', Image, queue_size=1000)
        # grasp_img_rgb_pub = rospy.Publisher('/grasp/grasp_img_rgb', Image, queue_size=1000)
        # 订阅
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depCallback)  # realsense的depth图像
        rospy.Subscriber("/camera/color/image_raw", Image, rgbCallback)  # realsense的rgb图像（抓取之前）
        rospy.Subscriber("/camera/color/image_raw", Image, rgb2Callback)  # realsense的rgb图像（抓取完成之后）
        rospy.Subscriber("/grasp/grasp_detect_run", Int8, runGraspCallback)  # 抓取检测命令
        rospy.Subscriber("/Capture_picture", String, iscapturecallback)  # 抓取检测命令

        #cv2.startWindowThread()
        bridge = CvBridge()

        ims_rgb = []
        ims_dep = []
        graspStateRGB = False
        graspStateD = False
        rgb2_flag = 0

        time.sleep(1)

        # 运行设备
        #device_name = "cuda:0"# if torch.cuda.is_available() else "cpu"
        device_name = 'cpu'
        print('device_name = ', device_name)
        device = torch.device(device_name)
        # SGDN模型路径
        grasp_model = '/path/to/your/workspace/src/kinova-ros/sim_grasp/scripts/ckpt/epoch_0064_acc_0.1957_.pth'
        # 初始化抓取检测器
        sgdn = SGDN(grasp_model, device=device_name)

        print('waiting for topic /grasp/grasp_detect_run')
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

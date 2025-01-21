#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx
@ File ：test-sgdn.py
@ IDE ：PyCharm
@ Function : sgdn测试类
"""

import cv2
import os
import torch
import time
import numpy as np
import sys
sys.path.append('/home/cvpr/robot_ws/src/kinova-ros/sim_grasp/scripts/models')
from models.loss import get_pred
from models.ggcnn2 import GGCNN2
from models.common import post_process_output
import skimage.transform as skt
from skimage.feature import peak_local_max



def imresize(image, size, interp="nearest"):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                    #   anti_aliasing=False,
                      mode="constant")

def depth2Gray(im_depth):
    """
    将深度图转至8位灰度图
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max
    return (im_depth * k + b).astype(np.uint8)

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

def arg_thresh(array, thresh):
    """
    获取array中大于thresh的二维索引
    :param array: 二维array
    :param thresh: float阈值
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs


def input_depth_resize(img, scale):
    """
    对图像进行修补、resize
    :param img: 深度图像, np.ndarray (h, w)
    :return: 直接输入网络的tensor, resize的尺度
    """
    # assert img.shape[0] >= input_size and img.shape[1] >= input_size, '输入深度图的尺寸必须大于等于{}*{}'.format(input_size, input_size)

    # resize成input_size
    # scale = input_size * 1.0 / img.shape[0]
    img = imresize(img, scale, interp="bilinear")

    # 归一化
    img = np.clip((img - img.mean()), -1, 1)

    # 调整顺序，和网络输入一致
    img = img[np.newaxis, np.newaxis, :, :]     # (1, 1, h, w)
    im_tensor = torch.from_numpy(img.astype(np.float32))  # np转tensor

    return im_tensor, scale



class SGDN:
    def __init__(self, model, device):
        # 加载模型
        print('>> loading SGDN')
        self.device = device
        self.net = GGCNN2(angle_cls=18)
        pretrained_dict = torch.load(model, map_location=torch.device(device))
        self.net.load_state_dict(pretrained_dict, strict=True)
        print('>> load done')


    def predict(self, img, device, mode, scale, thresh=0.3, peak_dist=1, angle_k=18, GRASP_WIDTH_MAX=0.1):
        """
        预测抓取模型
        :param img: 输入深度图 np.array (h, w)
        :param thresh: 置信度阈值
        :param peak_dist: 置信度筛选峰值
        :param angle_k: 抓取角分类数
        :return:
            pred_grasps: list([row, col, angle, width])  width单位为米
        """
        # print(img.shape)
        # 预测
        im_tensor, self.scale = input_depth_resize(img, scale)
        # print('torch.unique(im_tensor) = ', torch.unique(im_tensor))

        self.net.eval()
        with torch.no_grad():
            self.able_out, self.angle_out, self.width_out = get_pred(self.net, im_tensor.to(device))
            able_pred, angle_pred, width_pred = post_process_output(self.able_out, self.angle_out, self.width_out, GRASP_WIDTH_MAX)

            print('max graspable = ', np.max(able_pred))
            if mode == 'peak':
                # 置信度峰值 抓取点
                pred_pts = peak_local_max(able_pred, min_distance=peak_dist, threshold_abs=thresh)
            elif mode == 'max':
                # 置信度最大的点
                loc = np.argmax(able_pred)
                row = loc // able_pred.shape[0]
                col = loc % able_pred.shape[0]
                pred_pts = np.array([[row, col]])
            elif mode == 'all':
                # 超过阈值的所有抓取点
                pred_pts = arg_thresh(able_pred, thresh=thresh)
            else:
                raise ValueError

            # 绘制预测的抓取三角形
            pred_grasps = []
            for idx in range(pred_pts.shape[0]):
                row, col = pred_pts[idx]
                conf = able_pred[row, col]
                angle = angle_pred[row, col] * 1.0 / angle_k * np.pi  # 预测的抓取角弧度
                width = width_pred[row, col]    # 米
                row = int(row * 1.0 / self.scale)
                col = int(col * 1.0 / self.scale)
                pred_grasps.append([row, col, angle, width, conf])
            
            print('output grasp num: ', len(pred_grasps))
            
            pred_grasps = np.array(pred_grasps, dtype=np.float)
            # 对pred_grasps排序
            idxs = np.argsort(pred_grasps[:, 4] * -1)
            pred_grasps = pred_grasps[idxs]

        return pred_grasps


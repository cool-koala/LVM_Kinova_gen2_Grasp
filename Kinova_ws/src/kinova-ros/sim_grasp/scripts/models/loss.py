# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/31 18:58
@Auth ： 王德鑫
@File ：loss.py
@IDE ：PyCharm
@Function: Loss
"""
import math
import time
import torch
import torch.nn.functional as F


# def compute_loss(net, x, y_pts, y_ang, y_wid):
#     """
#     计算损失
#     params:
#         net: 网络
#         x:     网络输入图像   (batch, 1,   h, w)
#         y_pts: 抓取点标签图   (batch, 1,   h, w)
#         y_ang: 抓取角标签图   (batch, bin, h, w)
#         y_wid: 抓取宽度标签图 (batch, bin, h, w)
#     """

#     # 获取网络预测
#     able_pred, angle_pred, width_pred = net(x)         # shape 同上

#     # 置信度损失
#     able_pred = torch.sigmoid(able_pred)
#     able_loss = F.binary_cross_entropy(able_pred, y_pts)

#     # 抓取角损失
#     angle_pred = torch.sigmoid(angle_pred)
#     angle_loss = F.binary_cross_entropy(angle_pred, y_ang)

#     # 抓取宽度损失
#     width_pred = torch.sigmoid(width_pred)
#     width_loss = F.binary_cross_entropy(width_pred, y_wid)

#     return {
#         'loss': able_loss + angle_loss + width_loss,
#         'losses': {
#             'able_loss': able_loss,
#             'angle_loss': angle_loss,
#             'width_loss': width_loss,
#         },
#         'pred': {
#             'able': able_pred,  
#             'angle': angle_pred, 
#             'width': width_pred,   
#         }
#     }


def get_pred(net, xc):
    able_pred, angle_pred, width_pred = net(xc)
    
    able_pred = torch.sigmoid(able_pred)
    angle_pred = torch.sigmoid(angle_pred)
    width_pred = torch.sigmoid(width_pred)
    return able_pred, angle_pred, width_pred

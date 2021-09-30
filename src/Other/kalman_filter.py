#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt

if __name__=="__main__":
    ## 1.设计一个匀加速直线运动，以观测此运动
    X_real = np.mat(np.zeros((2, 100))) # 空矩阵，用于存放真实状态向量
    X_real[:, 0] = np.mat([[0.0], # 初始状态向量
                           [1.0]])
    a_real = 0.1# 真实加速度
    F = np.mat([[1.0, 1.0], # 状态转移矩阵
                [0.0, 1.0]])

    B = np.mat([[0.5], # 控制矩阵
                [1.0]])
    for i in range(99):
        X_real[:, i + 1] = F * X_real[:, i] + B * a_real # 计算真实状态向量
    X_real = np.array(X_real)
    fig = plt.figure(1)
    plt.grid()
    plt.title('real displacement')
    plt.xlabel('k (s)')
    plt.ylabel('x (m)')
    plt.plot(X_real[0, :])
    plt.show()
    fig = plt.figure(2)
    plt.grid()
    plt.title('real velocity')
    plt.xlabel('k (s)')
    plt.ylabel('v (m/s)')
    plt.plot(X_real[1, :])
    plt.show()
    X_real = np.mat(X_real)

    ## 2.建立传感器观测值
    z_t = np.mat(np.zeros((2, 100))) # 空矩阵，用于存放传感器观测值
    H = np.mat(np.zeros((2, 2)))
    H[0, 0], H[1, 1] = -1.0, -1.0
    noise = np.mat(np.random.randn(2,100)) # 加入位移方差为1，速度方差为1的传感器噪声
    R = np.mat([[1.0, 0.0], # 观测噪声的协方差矩阵
                [0.0, 1.0]])
    for i in range(100):
        z_t[:, i] = H * X_real[:, i] + noise[:, i]
    z_t = np.array(z_t)
    fig = plt.figure(3)
    plt.grid()
    plt.title('sensor displacement')
    plt.xlabel('k (s)')
    plt.ylabel('x (m)')
    plt.plot(z_t[0, :])
    plt.show()
    fig = plt.figure(4)
    plt.grid()
    plt.title('sensor velocity')
    plt.xlabel('k (s)')
    plt.ylabel('v (m/s)')
    plt.plot(z_t[1, :])
    plt.show()
    z_t = np.mat(z_t)
    
    ## 3.执行线性卡尔曼滤波
    Q = np.mat([[0.0001, 0.0], # 状态转移协方差矩阵，我们假设外部干扰很小，
                [0.0, 0.0001]])# 转移矩阵可信度很高
    # 建立一系列空序列用于储存结果
    X_update = np.mat(np.zeros((2, 100)))
    P_update = np.zeros((100, 2, 2))
    X_predict = np.mat(np.zeros((2, 100)))
    P_predict = np.zeros((100, 2, 2))
    #X_update[:, 0] = X_real[:, 0] # 初始状态向量
    #X_predict[:, 0] = X_real[:, 0] # 初始状态向量
    P_update[0, :, :] = np.mat([[1.0, 0.0], # 状态向量协方差矩阵初值
                                [0.0, 1.0]])
    P_predict[0, :, :] = np.mat([[1.0, 0.0], # 状态向量协方差矩阵初值
                                 [0.0, 1.0]])
    for i in range(99):
        # 预测
        X_predict[:, i + 1] = F * X_update[:, i] + B * a_real
        P_p = F * np.mat(P_update[i, :, :]) * F.T + Q
        P_predict[i + 1, :, :] = P_p
        # 更新
        K = P_p * H.T * np.linalg.inv(H * P_p * H.T + R) # 卡尔曼增益
        P_u = P_p - K * H * P_p
        P_update[i + 1, :, :] = P_u
        X_update[:, i + 1] = X_predict[:, i + 1] + K * (z_t[:, i + 1] - H * X_predict[:, i + 1])
    X_update = np.array(X_update)   
    X_real = np.array(X_real) 
    fig = plt.figure(5)
    plt.grid()
    plt.title('Kalman predict displacement')
    plt.xlabel('k (s)')
    plt.ylabel('x (m)')
    plt.plot(X_real[0, :], label='real', color='b')
    plt.plot(X_update[0, :], label='predict', color='r')
    plt.legend()
    plt.show()
    fig = plt.figure(6)
    plt.grid()
    plt.title('Kalman predict velocity')
    plt.xlabel('k (s)')
    plt.ylabel('v (m/s)')
    plt.plot(X_real[1, :], label='real', color='b')
    plt.plot(X_update[1, :], label='predict', color='r')
    plt.legend()
    plt.show()
    X_update = np.mat(X_update)
    X_real = np.mat(X_real)
    
    




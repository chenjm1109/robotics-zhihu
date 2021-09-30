#!/usr/bin/env python
# coding: utf-8

# In[1]:



import cv2
import numpy as np
import math
import time

def check_point(point, map):
    feasibility = True
    if map[point[1], point[0]] == 0:
        feasibility = False
    return feasibility

def check_path(point_current, point_other, map):
    # 路径检查的采样点数，取路径横向与纵向长度的较大值，保证每个像素都能验证到
    step_length = max(abs(point_current[0] - point_other[0]),
                      abs(point_current[1] - point_other[1]))
    path_x = np.linspace(point_current[0], point_other[0], step_length + 1)
    path_y = np.linspace(point_current[1], point_other[1], step_length + 1)
    for i in range(step_length + 1):
        if (not check_point([int(math.ceil(path_x[i])), int(math.ceil(path_y[i]))], map)):
            return False
    return True
    
def  straight_distance(point_a, point_b):
    distance = math.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)
    return distance

def A_star_list(vertex, 
             self_index, 
             goal_index, 
             parent_index,
             historic_cost):
    historic_cost = historic_cost
    heuristic_cost = straight_distance(vertex[self_index, :], 
                                            vertex[goal_index, :])
    total_cost = historic_cost + heuristic_cost
    A_list = np.mat([self_index, parent_index, total_cost])
    return A_list

def plot_img(map_img, vertex, path_state, close_list, state):
    # 画点
    point_size = 5
    point_color = (0, 127, 0) # BGR
    thickness = 4 
    vertex_tuple = tuple(map(tuple, vertex)) # 将数组转化为元组
    for point in vertex_tuple:
        cv2.circle(map_img, point, point_size, point_color, thickness)

    for i in range(vertex.shape[0]):
        for j in range(vertex.shape[0]):
            if path_state[i, j]==1:
                cv2.line(map_img,vertex_tuple[i],vertex_tuple[j],(255,150,150),2)

    
    # 回溯绘制最优路径
    point_a_index = 1
    if state:
        while point_a_index != 0:
            exist_index = int(np.argwhere(close_list[:,0]==point_a_index)[0,0])
            point_b_index = int(close_list[exist_index, 1])
            cv2.line(map_img,vertex_tuple[point_a_index],vertex_tuple[point_b_index],(0,0,255),4)
            point_a_index = point_b_index
    # 显示图片
    cv2.imshow("地图", map_img)# 转为RGB显示
    cv2.waitKey()

def PRM():
    time_start=time.time()
    print('开始PRM算法：')
    ## 预处理
    # 图像路径
    image_path = "/home/jimchan/Documents/motion_planning/map_1.bmp"
    # 读取图像
    img = cv2.imread(image_path)# np.ndarray BGR uint8
    # 灰度化
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret,img_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY) 
    

    # 起点坐标
    p_strat = np.array([10, 10])
    # 终点坐标
    p_goal = np.array([490, 490])
    # PRM算法采样点数量
    sample_num = 100
    
    # 是否显示过程点
    display = True
    # 初始化顶点集
    vertex = np.vstack((p_strat, p_goal))

    ## 构造地图
    # 采样并添加顶点
    while vertex.shape[0] < (sample_num + 2):
        # 随机采样
        x = np.random.randint(0, img_binary.shape[0] - 1, (1, 2))
        # 点碰撞检测，将合理点添加到顶点集
        if check_point(x[0, :], img_binary):
            vertex = np.vstack((vertex, x))
    
    ## 添加连接路径
    path_state = np.zeros((sample_num + 2,sample_num + 2))
    
    for i in range(sample_num + 2):
        for j in range(sample_num + 2):
            # 如果距离小于50且路径不碰撞
            if (straight_distance(vertex[i, :], vertex[j, :]) <= 1000.0) and (check_path(vertex[i, :], vertex[j, :], img_binary)):
                path_state[i, j] = 1 # 路径状态置为1
    
    ## A*搜索
    # 搜索起点，数组依次为：[顶点集，顶点集索引，目标点索引，父节点索引，历史代价]
    A_list = A_star_list(vertex, 0, 1, -1, 0)
    close_list = np.mat([[-1,-1,-1]])
    open_list = A_list
    # 如果开表非空
    while open_list.shape[0] > 0: 
        # 获取开表中代价最小点及其索引
        min_cost_index = np.argmin(open_list[:, 2])
        n_list = open_list[min_cost_index, :]
        # 将最小点转移到闭表
        close_list = np.append(close_list,open_list[min_cost_index, :],axis = 0)
        open_list = np.delete(open_list, min_cost_index, 0)
        # 如果不是目标点（通过索引判断）
        if n_list[0, 0] != 1:
            # 查找子节点的索引
            sub_list = np.array([], dtype = int)
            for i in range(sample_num + 2):
                if path_state[int(n_list[0, 0]), i] == 1:
                    sub_list = np.append(sub_list, i)
            for i in sub_list: # i为顶点集索引
                # 如果子节点在开表中, 比较并更新
                if i in open_list[:, 0]:
                    # 获取开表中已有点的索引
                    exist_index = np.argwhere(open_list[:,0]==i)[0,0]
                    # 比较代价，若新点更小，则替换
                    A_list = A_star_list(vertex, i, 1, n_list[0, 0], n_list[0, 2])
                    if A_list[0, 2] < open_list[exist_index, 2]:
                        open_list[exist_index, :] = A_list[0, :]
                
                # 如果子节点不在开表与闭表中，将其加入开表
                elif not ((i in open_list[:, 0]) or (i in close_list[:, 0])):
                    A_list = A_star_list(vertex, i, 1, n_list[0, 0], n_list[0, 2])
                    open_list = np.append(open_list,A_list,axis = 0)
        else:
            print('找到最优路径，结束搜索')
            time_end=time.time()
            print('总耗时')
            print(time_end-time_start)
            plot_img(img, vertex, path_state, close_list, True)
            return close_list, path_state, vertex

    else:
        print('没有找到合理的路径')
        time_end=time.time()
        print('总耗时')
        print(time_end-time_start)
        plot_img(img, vertex, path_state, close_list, False)
        return close_list, path_state, vertex
    

if __name__ == '__main__':
    close_list, path_state, vertex = PRM()

    
    


# In[269]:





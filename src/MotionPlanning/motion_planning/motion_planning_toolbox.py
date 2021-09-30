#coding=utf-8
import cv2
import numpy as np
import math

## 运动规划工具箱

def maxmin_normalization(x, lower_boundary, upper_boundary):
    # 映射函数
    x_max = np.max(x)
    x_min = np.min(x)
    x = (x - x_min) / (x_max - x_min) * (upper_boundary - lower_boundary)
    return x

def check_point(point, map_img):
    ## 点的碰撞检测工具
    point = np.mat(point)
    feasibility = True
    if (int(point[:, 0]) < map_img.shape[0] and 
        int(point[:, 1]) < map_img.shape[1] and 
        int(point[:, 0]) >= 0 and 
        int(point[:, 1]) >= 0):
        if map_img[int(point[:, 1]), int(point[:, 0])] == 0:
            feasibility = False
    else:
        feasibility = False
    return feasibility

def check_path(point_current, point_other, map_img):
    ## 路径检查的采样点数，取路径横向与纵向长度的较大值，保证每个像素都能验证到
    step_length = max(abs(point_current[0, 0] - point_other[0, 0]),
                      abs(point_current[0, 1] - point_other[0, 1]))
    path_x = np.linspace(point_current[0, 0], point_other[0, 0], step_length + 1)
    path_y = np.linspace(point_current[0, 1], point_other[0, 1], step_length + 1)
    for i in range(int(step_length + 1)):
        if (not check_point([int(math.ceil(path_x[i])), int(math.ceil(path_y[i]))], map_img)):
            return False
    return True

def straight_distance(point_a, point_b):
    ## 直线距离计算工具，point_a可以是矩阵形式的点集，将返回一个矩阵，每行对应各点与point_b的直线距离
    distance = np.sqrt(np.sum(np.multiply(point_a - point_b, point_a - point_b), axis=1))
    return distance

def A_star_list(vertex, 
             self_index, 
             goal_index, 
             parent_index,
             historic_cost):
    ## A*算法的路径代价表计算工具
    historic_cost = historic_cost
    heuristic_cost = straight_distance(vertex[self_index, :], 
                                            vertex[goal_index, :])
    total_cost = historic_cost + heuristic_cost
    A_list = np.mat([self_index, parent_index, total_cost])
    return A_list

def A_star_algorithm(vertex, adjacency_mat, start_index, goal_index):
    ## A*搜索算法
    num_sample = vertex.shape[0] - 2
    # 搜索起点，数组依次为：[顶点集，顶点集索引，目标点索引，父节点索引，历史代价]
    A_list = A_star_list(vertex, start_index, goal_index, -1, 0)
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
        if n_list[0, 0] != goal_index:
            # 查找子节点的索引
            sub_list = np.array([], dtype = int)
            for i in range(num_sample + 2):
                if adjacency_mat[int(n_list[0, 0]), i] == 1:
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
            return vertex, adjacency_mat, close_list, True

    else:
        print('没有找到合理的路径')
        return vertex, adjacency_mat, close_list, False

def A_star_plot(map_img, vertex, adjacency_mat, close_list):
    ## 绘制A*求解结果图
    # 画点
    point_size = 5
    point_color = (0, 127, 0) # BGR
    thickness = 4 
    # 将矩阵转化为数组，再转化为元组，以供cv2使用
    vertex = np.array(vertex)
    vertex_tuple = tuple(map(tuple, vertex)) 
    for point in vertex_tuple:
        cv2.circle(map_img, point, point_size, point_color, thickness)

    for i in range(vertex.shape[0]):
        for j in range(vertex.shape[0]):
            if adjacency_mat[i, j]==1:
                cv2.line(map_img, vertex_tuple[i], vertex_tuple[j], (255,150,150), 2)

    
    # 回溯绘制最优路径
    point_a_index = 1
    while point_a_index != 0:
        exist_index = int(np.argwhere(close_list[:,0]==point_a_index)[0,0])
        point_b_index = int(close_list[exist_index, 1])
        cv2.line(map_img,vertex_tuple[point_a_index],vertex_tuple[point_b_index],(0,0,255),4)
        point_a_index = point_b_index
    # 显示图片
    cv2.imshow("地图", map_img)# 转为RGB显示
    cv2.waitKey()
    
def tree_plot(map_img, rrt_tree):
    ## 绘制树形图
    # 画点
    point_size = 5
    point_color = (0, 127, 0) # BGR
    thickness = 4
    # 将矩阵转化为数组并转为整型，再转化为元组，以供cv2使用   
    vertex = np.around(np.array(rrt_tree)).astype(int)
    vertex_tuple = tuple(map(tuple, vertex)) 
    # 画点画线
    for point in vertex_tuple:
        cv2.circle(map_img, point[0 : 2], point_size, point_color, thickness)
        if point[0] != 0:
            cv2.line(map_img, point[0 : 2], vertex_tuple[point[2]][0 : 2], (255,150,150), 2)
    # 回溯绘制最优路径
    # 找到与目标点最近的点
    point_a_index = -1
    while point_a_index != 0:
        point_b_index = int(rrt_tree[point_a_index, 2])
        cv2.line(map_img,vertex_tuple[int(point_a_index)][0 : 2], vertex_tuple[int(point_b_index)][0 : 2],(0,0,255),4)
        point_a_index = point_b_index
    cv2.imshow("地图", map_img)# 转为RGB显示
    cv2.waitKey()

def potential_plot(map_original, img_potential, point_road):
    point_road_trans = point_road * 0.0
    point_road_trans[:, 0] = point_road[:, 1]
    point_road_trans[:, 1] = point_road[:, 0]
    point_road = point_road_trans
    # 整理势场图
    img = map_original # 使用原图绘图
    #img_potential[img_potential > 255.0] = 255.0
    #img_potential = maxmin_normalization(img_potential, 0, 255)
    #img_potential = img_potential.astype(int)
    #img = np.zeros([img_potential.shape[0],img_potential.shape[0],1],np.uint8)
    #img[:,:,0] = img_potential[:,:]
    # 整理路线点集
    vertex = np.around(np.array(point_road)).astype(int)
    vertex_tuple = tuple(map(tuple, vertex))
    for i in range(point_road.shape[0] - 1):
      cv2.line(img, vertex_tuple[i + 1], vertex_tuple[i], (0,0,255), 1)
    cv2.namedWindow("地图",0);
    cv2.resizeWindow("地图", 500, 500);
    cv2.imshow("地图", img)# 转为RGB显示
    cv2.waitKey()
    cv2.destroyAllWindows()
    












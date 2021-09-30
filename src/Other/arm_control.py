#!/usr/bin/env python
#coding=utf-8
'''
The kinematics solver for 6-DOF manipulator on pigot.

Name: Jim CHAN
Email:522706601@qq.com

'''

import numpy as np
import math

def pos2trans(pos):
    ## Transform the pose vector into homogeneous transformation matrix.
    px = pos[0, 0]
    py = pos[0, 1]
    pz = pos[0, 2]
    R = pos[0, 3]
    P = pos[0, 4]
    Y = pos[0, 5]
    rotx = np.mat([[ 1.0,       0.0,         0.0,           0.0],
                   [ 0.0,       math.cos(R), -math.sin(R),  0.0],
                   [ 0.0,       math.sin(R),  math.cos(R),  0.0],
                   [ 0.0,       0.0,         0.0,           1.0]])
    
    roty = np.mat([[ math.cos(P),  0.0,      -math.sin(P),  0.0],
                   [ 0.0,          1.0,       0.0,          0.0],
                   [ math.sin(P),  0.0,       math.cos(P),  0.0],
                   [ 0.0,          0.0,       0.0,          1.0]])
    
    rotz = np.mat([[ math.cos(Y), -math.sin(Y),  0.0,       0.0],
                   [ math.sin(Y),  math.cos(Y),  0.0,       0.0],
                   [ 0.0,          0.0,          1.0,       0.0],
                   [ 0.0,          0.0,          0.0,       1.0]])
    trans_pos = np.mat([[ 1.0,  0.0,  0.0,  px  ],
                        [ 0.0,  1.0,  0.0,  py  ],
                        [ 0.0,  0.0,  1.0,  pz  ],
                        [ 0.0,  0.0,  0.0,  1.0 ]])
    trans_mat = trans_pos * rotx * roty * rotz
    return trans_mat

def trans2pos(trans):
    ## Transform the homogeneous transformation matrix into the pose vector.
    pos = np.mat([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pos[0, 0] = trans[0, 3]
    pos[0, 1] = trans[1, 3]
    pos[0, 2] = trans[2, 3]
    pos[0, 3] = math.atan2(trans[2, 1], trans[2, 2])
    pos[0, 4] = math.atan2(-trans[2, 0], math.sqrt(trans[2, 1]**2 + trans[2, 2]**2))
    pos[0, 5] = math.atan2(trans[1, 0], trans[0, 0])
    return pos
    

def fkine(theta):
    ## Forward kinematics
    ## Input the joint angle matrix (rad).
    link_serial = np.mat([[0.00,  0.00,  0.00, 0.00],
                          [0.00, -1.57,  0.00, 0.00],
                          [0.60,  0.00,  0.00, 0.00],
                          [0.00, -1.57,  0.64, 0.00],
                          [0.00,  1.57,  0.00, 0.00],
                          [0.00, -1.57,  0.00, 0.00]])
    link_serial[:, 3] = theta.T
    n = link_serial.shape[0]
    fkine_trans = np.mat(np.eye(4))
    for i in range(n):
        a = link_serial[i, 0]
        sa = math.sin(link_serial[i, 1])
        ca = math.cos(link_serial[i, 1])
        d = link_serial[i, 2]
        st = math.sin(link_serial[i, 3])
        ct = math.cos(link_serial[i, 3])
        T = np.mat([[ ct,     -st,      0.0,   a     ],
                    [ st*ca,   ct*ca,  -sa,   -d*sa  ],
                    [ st*sa,   ct*sa,   ca,    d*ca  ],
                    [ 0.0,     0.0,     0.0,   1.0   ]])
        fkine_trans = fkine_trans * T
    fkine_pos = trans2pos(fkine_trans)
    return fkine_pos
    
def ikine(pos):
    ## Inverse kinematics
    ## Input the pose vector.
    trans_mat = pos2trans(pos)
    theta = np.mat([0.00,  0.00,  0.00,  0.00,  0.00,  0.00]) # Just to get the DH-table here, we can input any value.
    link_serial = DH_param(theta)
    a2 = link_serial[2, 0]
    d4 = link_serial[3, 2]
    nx = trans_mat[0, 0]
    ny = trans_mat[1, 0]
    nz = trans_mat[2, 0]
    ox = trans_mat[0, 1]
    oy = trans_mat[1, 1]
    oz = trans_mat[2, 1]
    ax = trans_mat[0, 2]
    ay = trans_mat[1, 2]
    az = trans_mat[2, 2]
    px = trans_mat[0, 3]
    py = trans_mat[1, 3]
    pz = trans_mat[2, 3]
    s3 = (px**2 + py**2 + pz**2 - d4**2 - a2**2) / ( -2 * a2 * d4)
    c3 = math.sqrt(1 - s3**2)
    # c3 = - math.sqrt(1 - s3**2)

    theta_1 = math.atan2(py, px)
    theta_3 = math.atan2(s3, c3)
    s1 = math.sin(theta_1)
    c1 = math.cos(theta_1)
    s23 = (-a2*c3*pz + (c1*px+s1*py)*(a2*s3-d4))/(pz**2 + (c1*px+s1*py)**2)
    c23 = ((-d4+a2*s3)*pz-(c1*px+s1*py)*(-a2*c3))/(pz**2 + (c1*px+s1*py)**2)
    
    theta_23 = math.atan2(s23, c23)
    print(theta_23)
    theta_2 = theta_23 - theta_3
    s4_m_s5 = -ax*s1 + ay*c1
    c4_m_s5 = -ax*c1*c23 - ay*s1*c23 + az*s23
#    print(s4_m_s5)
#    print(c4_m_s5)
    # Verification singularity
    if abs(s4_m_s5)>0.01 or abs(c4_m_s5)>0.01:
        theta_4 = math.atan2(s4_m_s5, c4_m_s5)
    else:
        theta_4 = 0
    s4 = math.sin(theta_4)
    c4 = math.cos(theta_4)
    s5 = -ax*(c1*c23*c4+s1*s4)-ay*(s1*c23*c4-c1*s4)+az*s23*c4
    c5 = -ax*c1*s23-ay*s1*s23-az*c23
    theta_5 = math.atan2(s5, c5)
    s6 = -nx*(c1*c23*s4-s1*c4) - ny*(s1*c23*s4+c1*c4) + nz*s23*s4
    c6 = nx*((c1*c23*c4+s1*s4)*c5-c1*s23*s5) + ny*((s1*s23*c4-c1*s4)*c5-s1*s23*s5)-nz*(s23*c4*c5+c23*s5)
    theta_6 = math.atan2(s6, c6)
    theta_4 = theta_4 - math.pi
    theta_5 = -theta_5
    ikine_theta = np.mat([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
    return ikine_theta
     
def DH_param(theta):
    ## Get the DH-table.
    link_serial = np.mat([[0.00,  0.00,       0.00, 0.00],
                          [0.00, -math.pi/2,  0.00, 0.00],
                          [0.60,  0.00,       0.00, 0.00],
                          [0.00, -math.pi/2,  0.64, 0.00],
                          [0.00,  math.pi/2,  0.00, 0.00],
                          [0.00, -math.pi/2,  0.00, 0.00]])
    link_serial[:, 3] = theta.T
    return link_serial
    
if __name__ == '__main__' :
    theta = np.mat([0.0,  1.5,  0.5,  0.00,  0.00,  0.00]) # joint angle (rad)
    pos = np.array([ 0.6,  0.000,  -0.64,  0.0,  0.5,  0.5]) # end pose vector (m)
    fkine_pos = fkine(theta)
    ikine_theta = ikine(fkine_pos)
    print('The forward solution is:')
    print(fkine_pos)
    print('The inverse solution is:')
    print(ikine_theta)



import numpy as np
import os
import sys


def load_xyz(xyz_file):
    xyz = np.zeros((128,3))
    channel_name = []
    f = open(xyz_file, 'r')
    lines = f.readlines()
    lines = lines[1:]
    ratio = 1000
    for i in range(len(lines)):
        line = lines[i]
        xyzc = line.split()
        xyz[i,0] = float(xyzc[0])/ratio
        xyz[i,1] = float(xyzc[1])/ratio
        xyz[i,2] = float(xyzc[2])/ratio
        channel_name.append(xyzc[3])
    
    #print(xyz)
    #print(channel_name)
    return xyz, channel_name

def map_function(xyz):
    X = xyz[:,0]/(1+xyz[:,2]*3)
    Y = xyz[:,1]/(1+xyz[:,2]*3)
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    
    X_center = (np.max(X) + np.min(X))/2
    Y_center = (np.max(Y) + np.min(Y))/2

    X = X - X_center
    Y = Y - Y_center

    X_scale = np.max(X)
    Y_scale = np.max(Y)

    X = X / X_scale *1.02
    Y = Y / Y_scale *1.02
    
    XY = np.concatenate((X, Y), axis=-1)
    #center = np.mean(XY, axis=0)
    #XY_decenter = XY - center
    #print(X.shape, Y.shape, XY.shape)
    return XY

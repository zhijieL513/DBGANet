from torch.utils.data import Dataset
import numpy as np
import glob
import trimesh
import pandas as pd
import os
import vedo
import json
import torch


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="0,2,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m


class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, labels_dir, num_classes=15, patch_size=7000):
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.labels_dir = labels_dir

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mesh_file = self.data_list.iloc[idx][0]
        points_normals = np.loadtxt(mesh_file) # points_normals: (n, 6)
        
        points = points_normals[:, :3]
        normals = points_normals[:, 3:]
        
        
        points_normals = np.concatenate((points, normals), axis=-1) # points_normals: (n, 6)

        # 处理标签
        label_file = glob.glob(
            os.path.join(self.labels_dir, f'**/{os.path.basename(mesh_file).replace(".obj", ".txt")}')) + glob.glob(
            os.path.join(self.labels_dir, f'{os.path.basename(mesh_file).replace(".obj", ".txt")}'))
        label_file = label_file[0]
        labels = np.loadtxt(label_file) # points_normals: (n, )
        labels = labels.reshape(-1,1) # labels:(n, 1)
        
        # 归一化
        points_normals[:, 0:3], c, m = pc_normalize(points_normals[:, 0:3])
        points_normals[:, 3:6], _, _ = pc_normalize(points_normals[:, 3:6])

        
        # 归一化之后的坐标
        points = points_normals[:, :3]
        
        # 求质心坐标和标签
        temp_lab = labels[:, 0]
        lab_tooth_dict = {}
        for i in range(15):
            lab_tooth_dict[i] = []
        for i, lab in enumerate(temp_lab):
            lab_tooth_dict[lab].append(list(points[i]))
        barycenter = np.zeros([15, 3])
        for k, v in lab_tooth_dict.items():
            if v == []:
                continue
            temp = np.array(lab_tooth_dict[k])
            barycenter[k] = temp.mean(axis=0)
        barycenter_label = np.zeros([15,])
        for i, j in enumerate(barycenter_label):
            barycenter_label[i] = 1
            if barycenter[i][0]==0 and barycenter[i][1]==0 and barycenter[i][2]==0: 
                barycenter_label[i] = 0
        barycenter_label = barycenter_label[1:]
        barycenter = barycenter[1:]
            
        barycenter_label = barycenter_label.reshape(-1,1) # (15, 1)

        X = points_normals.transpose(1, 0) # （6, n）
        Y = labels.transpose(1, 0) # (1, n)
        barycenter_label = barycenter_label.transpose(1, 0) # (1, 15)
        sample = {'cells': torch.from_numpy(X), 'labels': torch.from_numpy(Y),'mesh_file': mesh_file, 'c': c, 'm': m, 'barycenter': torch.from_numpy(barycenter), 'barycenter_label': torch.from_numpy(barycenter_label)}
        return sample
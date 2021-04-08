import torch
import torch.nn as nn
import os
import sys
import time
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import copy

class dataGener(Dataset):
    def __init__(self, pose_txt, map_file, radar_dir, pose_num, d_xyt,\
         img_res, xySizes, radar_size, img_batch, P_portion, device, odom_file):
        print('Data init')

        self.radar_dir = radar_dir
        self.pose_num = pose_num
        self.d_xyt = d_xyt
        self.img_res = img_res
        self.xySizes = xySizes
        self.radar_size = radar_size
        self.img_batch = img_batch
        self.sub_map_size = int(1.5*radar_size)
        self.P_portion = P_portion

        # poses
        self.pose = np.zeros((pose_num,3))
        pose_matrix = open(pose_txt).read()
        pose_matrix = np.array([item.split() for item in pose_matrix.split('\n')[:-1]])
        for i in range(pose_num):
            self.pose[i,0] = pose_matrix[i,0]
            self.pose[i,1] = pose_matrix[i,1]
            # self.pose[i,2] = 180 * float(pose_matrix[i,2]) / np.pi
            self.pose[i,2] = pose_matrix[i,2]
        self.pose = np.around(self.pose, decimals=5)
        # to torch tensor
        self.pose = torch.from_numpy(self.pose).float().to(device)

        # ego-motion from GAN
        # for more robust particle filter
        # motion_cnt = pose_cnt - 1
        motion_num = pose_num - 1
        self.motion = np.zeros((motion_num,3))
        motion_matrix = open(odom_file).read()
        motion_matrix = np.array([item.split() for item in motion_matrix.split('\n')[:-1]])
        for i in range(motion_num):
            self.motion[i,0] = motion_matrix[i,0]
            self.motion[i,1] = motion_matrix[i,1]
            self.motion[i,2] = motion_matrix[i,2]
        # to torch tensor
        self.motion = torch.from_numpy(self.motion).float().to(device)

        # global map
        self.global_map = Image.open(map_file)

        affine_tensor = torch.zeros(img_batch,2,3).to(device)

        cnt = 0
        for i in range(self.P_portion):
            for j in range(self.P_portion):
                for k in range(self.P_portion):
                    # every sample
                    dx = float((d_xyt[0,i]*2/self.img_res)/self.radar_size)
                    dy = float((d_xyt[1,j]*2/self.img_res)/self.radar_size)
                    dt = -1*float(d_xyt[2,k]) * np.pi / 180
                    affine_tensor[cnt] = torch.tensor([[math.cos(dt), -math.sin(dt), -dy],
                                                        [math.sin(dt), math.cos(dt), -dx]], dtype=torch.float)
                    cnt += 1

        grid_batch_size = (self.img_batch, 1, self.radar_size, self.radar_size)
        self.grid_tensor = F.affine_grid(affine_tensor, grid_batch_size)

    def get_data(self, pose_, iter_, device):

        # do not change
        # pose_lidar = copy.deepcopy(pose_)
        pose_lidar = pose_.view(-1).cpu().detach().numpy()

        # gen radar tensor
        # radar_img_id = pose_id + 1 (pose_id from zero)
        radar_img = Image.open(self.radar_dir + str(int(iter_+1)) + '.png')
        data_radar = torchvision.transforms.ToTensor()(radar_img).to(device)
        data_radar = data_radar.unsqueeze(0).repeat(1,1,1,1)

        # the rad to deg for image rotate
        pose_lidar[2] = float(pose_lidar[2]) * 180 / np.pi

        # pose coord to image coord by me
        pose_u = int(math.floor(-pose_lidar[1]/self.img_res) - self.xySizes[0]/self.img_res - 1)
        pose_v = int(-1*math.floor(pose_lidar[0]/self.img_res) + self.xySizes[3]/self.img_res - 1)

        # crop the map
        # top_ = pose_v - self.radar_size//2
        # left_ = pose_u - self.radar_size//2
        # sub_map = TF.crop(self.global_map, top_, left_, self.radar_size, self.radar_size)

        top_ = pose_v - self.sub_map_size//2
        left_ = pose_u - self.sub_map_size//2
        sub_map = TF.crop(self.global_map, top_, left_, self.sub_map_size, self.sub_map_size)

        # rotate by the pose degree
        # default: sub map center
        rotated_sub_map = TF.rotate(sub_map, -pose_lidar[2])

        # to tenser
        map_tensor = torchvision.transforms.ToTensor()(rotated_sub_map).to(device)
        map_tensor = map_tensor.unsqueeze(0).repeat(1, 1, 1, 1)

        # crop again
        top_left = (self.sub_map_size - self.radar_size) // 2
        map_tensor = map_tensor[:,:,top_left:top_left+self.radar_size,top_left:top_left+self.radar_size]

        # lidar_temp = TF.to_pil_image(map_tensor[0].cpu())
        # lidar_temp.save('/home/yinhuan/temp/'+str(iter_)+'.png')

        # radar_temp = TF.to_pil_image(data_radar[0].cpu())
        # radar_temp.save('/home/yinhuan/temp/r_'+str(iter_)+'.png')

        # one radar, one lidar map
        return data_radar, map_tensor

    def grid_sample_sub(self, scan_feature, map_feature, device):

        scan_feature_batch = scan_feature.repeat(self.img_batch, 1, 1, 1)

        map_feature_batch = map_feature.repeat(self.img_batch, 1, 1, 1)

        # grid sample on the
        map_feature_batch_sampled = F.grid_sample(map_feature_batch, self.grid_tensor, mode='nearest')

        map_scan_diff = scan_feature_batch - map_feature_batch_sampled

        # map_scan_diff = torch.mul(scan_feature_batch, map_feature_batch_sampled)

        return map_scan_diff
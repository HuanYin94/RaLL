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

class dataGener_train(Dataset):
    def __init__(self, pose_txt, map_file, radar_dir, pose_num, d_xyt, r_xyt,\
         img_res, xySizes, radar_size, img_batch, P_portion, start_id, end_id, device):
        print('Data init')

        self.radar_dir = radar_dir
        self.pose_num = pose_num
        self.d_xyt = d_xyt
        self.r_xyt = r_xyt
        self.img_res = img_res
        self.xySizes = xySizes
        self.radar_size = radar_size
        self.img_batch = img_batch
        self.sub_map_size = int(1.5*radar_size) # no used in lite version
        self.P_portion = P_portion
        self.start_id = start_id
        self.end_id = end_id

        # poses
        self.pose = np.zeros((pose_num,3))
        pose_matrix = open(pose_txt).read()
        pose_matrix = np.array([item.split() for item in pose_matrix.split('\n')[:-1]])
        for i in range(pose_num):
            self.pose[i,0] = pose_matrix[i,0]
            self.pose[i,1] = pose_matrix[i,1]
            # self.pose[i,2] = 180 * float(pose_matrix[i,2]) / np.pi
            self.pose[i,2] = pose_matrix[i,2]

        # global map
        self.global_map = Image.open(map_file)

        affine_tensor = torch.zeros(img_batch,2,3).to(device)

        # grid sample on radar_size
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

    def get_random_data(self, device):
        # random radar id from zero
        radar_id = np.random.randint(low=self.start_id, high=self.end_id, size=1)
        # radar_id = 0
        # gen radar tensor
        radar_img = Image.open(self.radar_dir + str(int(radar_id+1)) + '.png')
        data_radar = torchvision.transforms.ToTensor()(radar_img).to(device)
        data_radar = data_radar.unsqueeze(0).repeat(1,1,1,1)

        # gen lidar-pose randomly in centain range
        # self-motion in robot-frame to world frame
        gt_xyt = np.zeros(3)
        gt_xyt[0] = self.r_xyt[0,0] + (self.r_xyt[0,1]-self.r_xyt[0,0])*np.random.random()
        gt_xyt[1] = self.r_xyt[1,0] + (self.r_xyt[1,1]-self.r_xyt[1,0])*np.random.random()
        gt_xyt[2] = self.r_xyt[2,0] + (self.r_xyt[2,1]-self.r_xyt[2,0])*np.random.random()

        gt_xyt_return = torch.from_numpy(-1*gt_xyt).to(device, dtype=torch.float)

        # frame transfer, and note the rad to deg for image rotate
        pose_lidar = copy.deepcopy(self.pose[radar_id,:])
        pose_lidar = np.reshape(pose_lidar, (3))
        # align with the affine matrix
        pose_lidar[2] = pose_lidar[2] + float(gt_xyt[2] * np.pi / 180)
        pose_lidar[0] = pose_lidar[0] + math.cos(pose_lidar[2])*gt_xyt[0] - math.sin(pose_lidar[2])*gt_xyt[1]
        pose_lidar[1] = pose_lidar[1] + math.sin(pose_lidar[2])*gt_xyt[0] + math.cos(pose_lidar[2])*gt_xyt[1]
        # to deg for rotation
        pose_lidar[2] = pose_lidar[2] * 180 / np.pi

        # pose coord to image coord by me
        pose_u = int(math.floor(-pose_lidar[1]/self.img_res) - self.xySizes[0]/self.img_res - 1)
        pose_v = int(-1*math.floor(pose_lidar[0]/self.img_res) + self.xySizes[3]/self.img_res - 1)

        # crop to submap size
        top_ = pose_v - self.sub_map_size//2
        left_ = pose_u - self.sub_map_size//2
        sub_map = TF.crop(self.global_map, top_, left_, self.sub_map_size, self.sub_map_size)

        # rotate by the pose degree
        # default: sub map center
        rotated_sub_map = TF.rotate(sub_map, -pose_lidar[2])

        # to tenser
        map_tensor = torchvision.transforms.ToTensor()(rotated_sub_map).to(device)
        map_tensor = map_tensor.unsqueeze(0).repeat(1, 1, 1, 1)

        # crop again to the radar size
        top_left = (self.sub_map_size - self.radar_size) // 2
        map_tensor = map_tensor[:,:,top_left:top_left+self.radar_size,top_left:top_left+self.radar_size]

        # one radar, one lidar map
        return data_radar, map_tensor, gt_xyt_return

    def grid_sample_sub(self, scan_feature, map_feature, device):

        scan_feature_batch = scan_feature.repeat(self.img_batch, 1, 1, 1)

        map_feature_batch = map_feature.repeat(self.img_batch, 1, 1, 1)

        # grid sample on the sub map
        map_feature_batch_sampled = F.grid_sample(map_feature_batch, self.grid_tensor, mode='nearest')

        # minus
        map_scan_diff = torch.sub(scan_feature_batch, map_feature_batch_sampled)

        # map_scan_diff = map_scan_diff[:,:,50:350,50:350]
        # print(map_scan_diff.size())

        # for i in range(int(math.pow(self.P_portion,3))):
        #     diff_temp = TF.to_pil_image(map_scan_diff[i].cpu())
        #     diff_temp.save('/home/yinhuan/temp/'+str(i)+'.png')

        # map_scan_diff_= torch.mean(map_scan_diff, (2,3)).view(-1)
        # print(map_scan_diff_)
        # v = torch.min(map_scan_diff_)
        # i = (map_scan_diff_ == v).nonzero()[0].item()
        # print('index: ', i)

        # radar_temp = TF.to_pil_image(scan_feature_batch[0].cpu())
        # radar_temp.save('/home/yinhuan/temp/radar.png')

        # lidar_temp = TF.to_pil_image(map_feature_batch[0].cpu())
        # lidar_temp.save('/home/yinhuan/temp/map.png')

        return map_scan_diff

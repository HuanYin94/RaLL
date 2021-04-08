import os
import torch
import torch.nn as nn

class doubleNet(nn.Module):
    def __init__(self, i_channel, o_channel, m_channel=None):
        super(doubleNet, self).__init__()

        if not m_channel:
            m_channel = o_channel

        self.net = nn.Sequential(
        nn.Conv2d(i_channel, m_channel, kernel_size=3, padding=1),
        nn.InstanceNorm2d(m_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(m_channel, o_channel, kernel_size=3, padding=1),
        nn.InstanceNorm2d(o_channel),
        nn.ReLU(inplace=True))

    def forward(self, input):
        return self.net(input)

class singleNet(nn.Module):
    def __init__(self, i_channel, o_channel):
        super(singleNet, self).__init__()

        self.net = nn.Sequential(
        nn.Conv2d(i_channel, o_channel, kernel_size=3, padding=1),
        nn.InstanceNorm2d(o_channel),
        nn.ReLU(inplace=True))

    def forward(self, input):
        return self.net(input)

class unet(nn.Module):
    def __init__(self, i_channel, o_channel):
        super(unet, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv_down_1 = doubleNet(i_channel, 8)
        self.conv_down_2 = doubleNet(8, 16)
        self.conv_down_3 = doubleNet(16, 32)
        self.conv_down_4 = doubleNet(32, 64)
        self.conv_down_5 = doubleNet(64, 64)

        self.tr_up_5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.tr_up_4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.tr_up_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.tr_up_2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_up_5 = doubleNet(128, 32, 64)
        self.conv_up_4 = doubleNet(64, 16, 32)
        self.conv_up_3 = doubleNet(32, 8, 16) 
        self.conv_up_2 = doubleNet(16, o_channel, 8)

    def forward(self, i):

        # down
        f_down_1 = self.conv_down_1(i)
        f_down_1p = self.pool(f_down_1)

        f_down_2 = self.conv_down_2(f_down_1p)
        f_down_2p = self.pool(f_down_2)

        f_down_3 = self.conv_down_3(f_down_2p)
        f_down_3p = self.pool(f_down_3)

        f_down_4 = self.conv_down_4(f_down_3p)
        f_down_4p = self.pool(f_down_4)

        # last
        f_down_5 = self.conv_down_5(f_down_4p)

        # up
        f_up_5t = self.tr_up_5(f_down_5)
        f_merge_5 = torch.cat([f_down_4, f_up_5t], dim=1)
        f_up_5 = self.conv_up_5(f_merge_5)

        f_up_4t = self.tr_up_4(f_up_5)
        f_merge_4 = torch.cat([f_down_3, f_up_4t], dim=1)
        f_up_4 = self.conv_up_4(f_merge_4)

        f_up_3t = self.tr_up_3(f_up_4)
        f_merge_3 = torch.cat([f_down_2, f_up_3t], dim=1)
        f_up_3 = self.conv_up_3(f_merge_3)

        f_up_2t = self.tr_up_2(f_up_3)
        f_merge_2 = torch.cat([f_down_1, f_up_2t], dim=1)
        f_up_2 = self.conv_up_2(f_merge_2)

        return f_up_2


class siamese_minus(nn.Module):
    def __init__(self):
        super(siamese_minus, self).__init__()

        # add pre-process from CoRL 2019 Masking by Moving
        self.radar_pre_nn = unet(1,8)

        self.mask_net = nn.Sequential(
        nn.Conv2d(8, 1, kernel_size=1),
        nn.Sigmoid())

        self.radar_nn = unet(1,8)
        self.lidar_nn = unet(1,8)
        self.feat_net_radar = singleNet(8,1)
        self.feat_net_lidar = singleNet(8,1)

    def forward(self, scan_batch, map_batch):

        # pre-process on scan
        scan_batch_ = self.radar_pre_nn(scan_batch)
        scan_batch_sigmoid = self.mask_net(scan_batch_)
        # mask
        scan_batch_mask = torch.mul(scan_batch_sigmoid, scan_batch)

        # feature for both scan & map
        scan_batch_feature = self.feat_net_radar(self.radar_nn(scan_batch_mask))
        map_batch_feature = self.feat_net_lidar(self.lidar_nn(map_batch))

        return scan_batch_sigmoid, scan_batch_mask, scan_batch_feature, map_batch_feature
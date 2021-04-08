import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class patch_net(nn.Module):
    def __init__(self, P_portion, K_portion):
        super(patch_net, self).__init__()

        self.K_portion = K_portion
        self.P_portion = P_portion

        # 07.03
        self.cnn_diff = nn.Sequential(
        # lenet 1: 32 to 16
        nn.Conv2d(pow(P_portion,3), pow(P_portion,3), kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(pow(P_portion,3)),
        nn.ReLU(inplace=True),
        # lenet 2: 16 to 8
        nn.Conv2d(pow(P_portion,3), pow(P_portion,3), kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(pow(P_portion,3)),
        nn.ReLU(inplace=True),
        # lenet 3: 8 to 4
        nn.Conv2d(pow(P_portion,3), pow(P_portion,3), kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(pow(P_portion,3)),
        nn.ReLU(inplace=True),
        # lenet 3: 4 to 1  = avg pool ...
        nn.Conv2d(pow(P_portion,3), pow(P_portion,3), kernel_size=4, stride=1, padding=0),
        nn.BatchNorm2d(pow(P_portion,3)),
        nn.ReLU(inplace=True))

    def forward(self, diff_batch):

        diff_cube = diff_batch.view(1, -1, diff_batch.size(2), diff_batch.size(3))

        k_size = diff_cube.size(2) // self.K_portion
        k_stride = k_size

        # use unfold as a kind of cnn
        diff_cube = F.unfold(diff_cube, kernel_size=k_size, dilation=1, stride=k_stride)
        # diff_origin_size: B, C*, L(how many kernels) for reshape
        B, C, L = diff_cube.size()
        diff_cube = diff_cube.permute(0,2,1)
        diff_cube = diff_cube.view(L, -1, k_size, k_size)

        diff_cube = self.cnn_diff(diff_cube)
        # print(diff_cube.size())
        avg_vector = torch.mean(diff_cube, dim=0).view(-1)

        return avg_vector

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import sys
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from loader.data_gener_gs import dataGener_train
from loss.offset_loss_exp import offset_loss
from network.patch_net_new import patch_net
from network.siamese_triunet_new import siamese_minus

# multiple radar scans as one real batch
img_batch = 343 # constant to dataset
# img_batch = 1331 # constant to dataset
# img_batch = 729 # constant to dataset
iter_train = 1000 # how many poses for training
num_epoch = 201

l_rate = 0.001
decay = 0.98
w_decay = 0.00001

# params
pose_num = 8865
# select on certain poses from zero
start_id = 0
end_id = 7580

# for RSL comparison
d_xyt = np.array([[-6, -4, -2, 0, 2, 4, 6],
                  [-6, -4, -2, 0, 2, 4, 6],
                  [-6, -4, -2, 0, 2, 4, 6],])
r_xyt = np.array([[-6, 6],
                  [-6, 6],
                  [-6, 6]])

# d_xyt = np.array([[-4, -3, -2, -1, 0, 1, 2, 3, 4],
#                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
#                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],])
# r_xyt = np.array([[-4, 4],
#                   [-4, 4],
#                   [-4, 4]])

# d_xyt = np.array([[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],])
# r_xyt = np.array([[-5, 5],
#                   [-5, 5],
#                   [-5, 5]])

img_res = 0.25
xySizes = np.array([-800, 1200, -500, 1500])
radar_size = 512
# radar_size = 256

# 183
pose_txt = 'data/Data/ablation_study/25_512/seq01/pose_xy_01.txt'
map_file = 'data/Data/ablation_study/25_512/oxford_laser_map.png'
radar_dir = 'data/Data/ablation_study/25_512/seq01/radar_scans/'
# pose_txt = 'data/Data/ablation_study/25_256/seq01/pose_xy_01.txt'
# map_file = 'data/Data/ablation_study/25_256/oxford_laser_map.png'
# radar_dir = 'data/Data/ablation_study/25_256/seq01/radar_scans/'
# 251
# pose_txt = '/home/yinhuan/Data/183/seqes/seq01/pose_xy_01.txt'
# map_file = '/home/yinhuan/Data/183/seqes/laser_map.png'
# radar_dir = '/home/yinhuan/Data/183/seqes/seq01/radar_scans/'
# 57
# pose_txt = '/home/yinhuan/data/seqes/seq01/pose_xy_01.txt'
# map_file = '/home/yinhuan/data/seqes/laser_map.png'
# radar_dir = '/home/yinhuan/data/seqes/seq01/radar_scans/'

P_portion = 7
# P_portion = 9
# P_portion = 11

K_portion = 16
# K_portion = 8

alpha = 5

# beta = 5
beta = 1

# date = '06.29_MCL_025_512'
date = '07.10_MCL_25_512_777_r7'

# pre_sia_model_path = 'data/radar_lite/models/07.05_pre_siamese_25_512/siamese_model_10.pth'
pre_sia_model_path = 'data/radar_lite/models/07.10_MCL_25_512_777_r6/siamese_model_200.pth'
pre_patch_model_path = 'data/radar_lite/models/07.10_MCL_25_512_777_r6/patch_model_200.pth'

if __name__ == "__main__":

    # device = 'cpu'
    # device = 'cuda:2'
    device = torch.cuda.current_device()

    data_gener = dataGener_train(pose_txt, map_file, radar_dir, pose_num, d_xyt, r_xyt,\
         img_res, xySizes, radar_size, img_batch, P_portion, start_id, end_id, device)

    # make folders
    if not os.path.exists('data/radar_lite/models/' + date):
        os.mkdir('data/radar_lite/models/' + date)
    if not os.path.exists('data/radar_lite/log/' + date):
        os.mkdir('data/radar_lite/log/' + date)

    # if not os.path.exists('/home/yinhuan/Data/183/radar_lite/models/' + date):
    #     os.mkdir('/home/yinhuan/Data/183/radar_lite/models/' + date)
    # if not os.path.exists('/home/yinhuan/Data/183/radar_lite/log/' + date):
    #     os.mkdir('/home/yinhuan/Data/183/radar_lite/log/' + date)

    # if not os.path.exists('/home/yinhuan/radar_lite/models/' + date):
    #     os.mkdir('/home/yinhuan/radar_lite/models/' + date)
    # if not os.path.exists('/home/yinhuan/radar_lite/log/' + date):
    #     os.mkdir('/home/yinhuan/radar_lite/log/' + date)

    print('Build Model')

    # load pre-trained siamese model

    # CAN try multiple GPU
    siamese_model = siamese_minus().to(device)
    siamese_model.load_state_dict(torch.load(pre_sia_model_path))
    siamese_model.train()

    # create patch model
    patch_model = patch_net(P_portion, K_portion).to(device)
    patch_model.load_state_dict(torch.load(pre_patch_model_path))
    patch_model.train()
    
    # loss
    offset_loss = offset_loss(P_portion, alpha, beta, d_xyt, device).to(device)

    # optimizer
    optimizer = torch.optim.Adam(list(siamese_model.parameters()) + list(patch_model.parameters()), lr=l_rate, weight_decay=w_decay)

    # board
    writer = SummaryWriter('data/radar_lite/log/' + date)
    # writer = SummaryWriter('/home/yinhuan/Data/183/radar_lite/log/' + date)
    # writer = SummaryWriter('/home/yinhuan/radar_lite/log/' + date)

    for epoch in range(num_epoch):
        print('epoch ' + str(epoch))

        # save all
        torch.save(siamese_model.state_dict(), \
            'data/radar_lite/models/' + date + '/%s_model_%d.pth' % ('siamese', epoch))
        torch.save(patch_model.state_dict(), \
            'data/radar_lite/models/' + date + '/%s_model_%d.pth' % ('patch', epoch))

        # torch.save(siamese_model.state_dict(), \
        #     '/home/yinhuan/Data/183/radar_lite/models/' + date + '/%s_model_%d.pth' % ('siamese', epoch))
        # torch.save(patch_model.state_dict(), \
        #     '/home/yinhuan/Data/183/radar_lite/models/' + date + '/%s_model_%d.pth' % ('patch', epoch))

        # torch.save(siamese_model.state_dict(), \
        #     '/home/yinhuan/radar_lite/models/' + date + '/%s_model_%d.pth' % ('siamese', epoch))
        # torch.save(patch_model.state_dict(), \
        #     '/home/yinhuan/radar_lite/models/' + date + '/%s_model_%d.pth' % ('patch', epoch))

        for iter_ in range(iter_train):

            t0 = time.time()

            scan_, map_, gt_xyt = data_gener.get_random_data(device)

            scan_pre, scan_mask, scan_feature, map_feature = siamese_model(scan_, map_)

            map_scan_diff = data_gener.grid_sample_sub(scan_feature, map_feature, device)

            avg_vector = patch_model(map_scan_diff)

            loss, est_xyt = offset_loss(gt_xyt, avg_vector)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # heavy print
            print('[%s %d: %d/%d] %s %f %s %f %s %s %s %s'\
                %('Patch Training',epoch,iter_, iter_train-1,'L-r:',l_rate,'Loss:',loss.data,\
                'GT:',str(gt_xyt.detach().cpu().numpy()),'EST:',str(est_xyt.detach().cpu().numpy())))

            # add loss and error to board
            writer.add_scalar(date+'/Loss', loss, epoch*iter_train + iter_)
            writer.add_scalar(date+'/MSE_x', torch.pow(est_xyt[0]-gt_xyt[0],2), epoch*iter_train + iter_)
            writer.add_scalar(date+'/MSE_y', torch.pow(est_xyt[1]-gt_xyt[1],2), epoch*iter_train + iter_)
            writer.add_scalar(date+'/MSE_t', torch.pow(est_xyt[2]-gt_xyt[2],2), epoch*iter_train + iter_)

            print('time: ', time.time() - t0)

        # learning rate reduction
        if epoch % 1 == 0:
            l_rate*=decay
            optimizer = torch.optim.Adam(list(siamese_model.parameters()) + list(patch_model.parameters()), lr=l_rate, weight_decay=w_decay)

    print('Finished')

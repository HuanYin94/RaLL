from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import sys
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random

sys.path.append("..")
from loader.data_gener_ekf   import dataGener
from network.patch_net_new import patch_net
from network.siamese_triunet_new import siamese_minus
from ekf_filter.ekf_filter_nn import ekf_filter

# multiple radar scans as one real batch
img_batch = 343 # constant to dataset
# img_batch = 1331 # constant to dataset
# img_batch = 729 # constant to dataset
# iter_train = 1000 # how many poses for training
num_epoch = 10100

# l_rate = 0.0001
# decay = 0.999 # no ?
l_rate = 0.0001
decay = 0.9996 # no ?
w_decay = 0.00001

# train on seq01
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

img_res = 0.25
xySizes = np.array([-800, 1200, -500, 1500])
radar_size = 512

# 183
# pose_txt = 'data/Data/ablation_study/25_512/seq01/pose_xy_01.txt'
# map_file = 'data/Data/ablation_study/25_512/oxford_laser_map.png'
# radar_dir = 'data/Data/ablation_study/25_512/seq01/radar_scans/'
# pose_txt = 'data/Data/ablation_study/25_256/seq01/pose_xy_01.txt'
# map_file = 'data/Data/ablation_study/25_256/oxford_laser_map.png'
# radar_dir = 'data/Data/ablation_study/25_256/seq01/radar_scans/'
# 251
pose_txt = '/home/yinhuan/Data/183/ablation_study/25_512/seq01/pose_xy_01.txt'
map_file = '/home/yinhuan/Data/183/ablation_study/25_512/oxford_laser_map.png'
radar_dir = '/home/yinhuan/Data/183/ablation_study/25_512/seq01/radar_scans/'
# 57
# pose_txt = '/home/yinhuan/data/seqes/seq01/pose_xy_01.txt'
# map_file = '/home/yinhuan/data/seqes/laser_map.png'
# radar_dir = '/home/yinhuan/data/seqes/seq01/radar_scans/'

odom_file = '/home/yinhuan/Data/183/odom/seq01_odom.txt'

P_portion = 7

K_portion = 16

alpha = 5

lamda = 10

# ekf filter motion var
R = np.diag([1.0,
             1.0,
             np.deg2rad(5.0)]) ** 2 # motion cov

# length for train
pose_len = 5

pre_sia_model_path = '/home/yinhuan/NFS/radar_lite/models/07.10_MCL_25_512_777_r7/siamese_model_200.pth'
pre_patch_model_path = '/home/yinhuan/NFS/radar_lite/models/07.10_MCL_25_512_777_r7/patch_model_200.pth'

# pre_sia_model_path = '/home/yinhuan/NFS/radar_lite/models/07.10_MCL_25_512_777_r5/siamese_model_200.pth'
# pre_patch_model_path = '/home/yinhuan/NFS/radar_lite/models/07.10_MCL_25_512_777_r5/patch_model_200.pth'

# pre_sia_model_path = '/home/yinhuan/Data/183/radar_lite/models/08.08_ekf/siamese_model_1995.pth'
# pre_patch_model_path = '/home/yinhuan/Data/183/radar_lite/models/08.08_ekf/patch_model_1995.pth'

date = '08.19_ekf_nn_last_try'

if __name__ == "__main__":

    # set print
    np.set_printoptions(precision=5)

    device = 'cuda:2'
    # device = torch.cuda.current_device()

    data_gener = dataGener(pose_txt, map_file, radar_dir, pose_num, d_xyt,\
         img_res, xySizes, radar_size, img_batch, P_portion, device, odom_file)

    # make folders
    # if not os.path.exists('data/radar_lite/models/' + date):
    #     os.mkdir('data/radar_lite/models/' + date)
    # if not os.path.exists('data/radar_lite/log/' + date):
    #     os.mkdir('data/radar_lite/log/' + date)

    if not os.path.exists('/home/yinhuan/Data/183/radar_lite/models/' + date):
        os.mkdir('/home/yinhuan/Data/183/radar_lite/models/' + date)
    if not os.path.exists('/home/yinhuan/Data/183/radar_lite/log/' + date):
        os.mkdir('/home/yinhuan/Data/183/radar_lite/log/' + date)

    # if not os.path.exists('/home/yinhuan/radar_lite/models/' + date):
    #     os.mkdir('/home/yinhuan/radar_lite/models/' + date)
    # if not os.path.exists('/home/yinhuan/radar_lite/log/' + date):
    #     os.mkdir('/home/yinhuan/radar_lite/log/' + date)

    print('Load Model')

    # CAN try multiple GPU
    siamese_model = siamese_minus().to(device)
    siamese_model.load_state_dict(torch.load(pre_sia_model_path))
    siamese_model.train()

    # create patch model
    patch_model = patch_net(P_portion, K_portion).to(device)
    patch_model.load_state_dict(torch.load(pre_patch_model_path))
    patch_model.train()

    # optimizer
    optimizer = torch.optim.Adam(list(siamese_model.parameters()) + list(patch_model.parameters()), lr=l_rate, weight_decay=w_decay)

    # board
    # writer = SummaryWriter('data/radar_lite/log/' + date)
    writer = SummaryWriter('/home/yinhuan/Data/183/radar_lite/log/' + date)
    # writer = SummaryWriter('/home/yinhuan/radar_lite/log/' + date)

    # in this e2e paper, every epoch train one path
    for epoch in range(num_epoch):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   epoch ' + str(epoch))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # not save all
        if epoch % 100 == 0:
            # save all
            # torch.save(siamese_model.state_dict(), \
            #     'data/radar_lite/models/' + date + '/%s_model_%d.pth' % ('siamese', epoch))
            # torch.save(patch_model.state_dict(), \
            #     'data/radar_lite/models/' + date + '/%s_model_%d.pth' % ('patch', epoch))

            torch.save(siamese_model.state_dict(), \
                '/home/yinhuan/Data/183/radar_lite/models/' + date + '/%s_model_%d.pth' % ('siamese', epoch))
            torch.save(patch_model.state_dict(), \
                '/home/yinhuan/Data/183/radar_lite/models/' + date + '/%s_model_%d.pth' % ('patch', epoch))

            # torch.save(siamese_model.state_dict(), \
            #     '/home/yinhuan/radar_lite/models/' + date + '/%s_model_%d.pth' % ('siamese', epoch))
            # torch.save(patch_model.state_dict(), \
            #     '/home/yinhuan/radar_lite/models/' + date + '/%s_model_%d.pth' % ('patch', epoch))

        # random a start
        random_start = random.randint(start_id, end_id-pose_len)
        # random_start = 6654

        # init is needed
        # init at the previous pose since the next step is moion model
        init_id = random_start-1
        pose_init = data_gener.pose[init_id,:]

        # x_rand = ((6-(-6))*torch.rand(1) - 6).to(device, dtype=torch.float)
        # y_rand = ((6-(-6))*torch.rand(1) - 6).to(device, dtype=torch.float)
        # t_rand = ((0.104-(-0.104))*torch.rand(1) - 0.104).to(device, dtype=torch.float)

        # pose_init[0] = pose_init[0] + x_rand
        # pose_init[1] = pose_init[1] + y_rand
        # pose_init[2] = pose_init[2] + t_rand

        # init ekf localization 
        ekf = ekf_filter(pose_init, R, P_portion, d_xyt, lamda, device).to(device)

        # clear loss
        total_loss = torch.tensor(0).to(device, dtype=torch.float)
        pose_cnt = torch.tensor(0).to(device, dtype=torch.float)

        for iter_ in range(random_start, random_start+pose_len):
            print('------------------------------------------------ ', iter_)

            t0 = time.time()

            # motion for the previous pose
            # add noise on motions for training
            # # -3 ~ 3 / 0.0524
            # motion_ = data_gener.motion[iter_-1,:].clone()
            # x_rand = ((3-(-3))*torch.rand(1) - 3).to(device, dtype=torch.float)
            # y_rand = ((3-(-3))*torch.rand(1)- 3).to(device, dtype=torch.float)
            # t_rand = ((0.0524-(-0.0524))*torch.rand(1) - 0.0524).to(device, dtype=torch.float)

            # # -6 ~ 6 / 0.104
            # motion_ = data_gener.motion[iter_-1,:].clone()
            # x_rand = ((6-(-6))*torch.rand(1) - 6).to(device, dtype=torch.float)
            # y_rand = ((6-(-6))*torch.rand(1) - 6).to(device, dtype=torch.float)
            # t_rand = ((0.104-(-0.104))*torch.rand(1) - 0.104).to(device, dtype=torch.float)

            # -1 ~ 1 / 0.0524
            # motion_ = data_gener.motion[iter_-1,:].clone()
            # x_rand = ((1-(-1))*torch.rand(1) - 1).to(device, dtype=torch.float)
            # y_rand = ((1-(-1))*torch.rand(1) - 1).to(device, dtype=torch.float)
            # t_rand = ((0.0524-(-0.0524))*torch.rand(1) - 0.0524).to(device, dtype=torch.float)

            # motion_[0] = motion_[0] + x_rand
            # motion_[1] = motion_[1] + y_rand
            # motion_[2] = motion_[2] + t_rand

            # # do motion
            # ekf.doMotion(motion_)
            # print('motion: ', motion_)

            # motion for the previous pose
            ekf.doMotion(data_gener.motion[iter_-1,:])
            print('motion: ', data_gener.motion[iter_-1,:])

            # net
            scan_, map_ = data_gener.get_data(ekf.est_xyt, iter_, device)
            scan_pre, scan_mask, scan_feature, map_feature = siamese_model(scan_, map_)
            map_scan_diff = data_gener.grid_sample_sub(scan_feature, map_feature, device)
            avg_vector = patch_model(map_scan_diff)

            # do observation
            loss, est_xyt = ekf(data_gener.pose[iter_,:], avg_vector)

            # optimizer.zero_grad()

            # if pose_cnt == pose_len-1:
            #     loss.backward()
            # else:
            #     loss.backward(retain_graph=True)

            # optimizer.step()

            # effective
            pose_cnt = pose_cnt + 1
            # add
            total_loss = total_loss + loss

            # print
            g = data_gener.pose[iter_,:].view(-1).cpu().detach().numpy()
            e = est_xyt.view(-1).cpu().detach().numpy()
            print('gt: ', g, ' est: ', e)
            error_ = (g - e)
            error_[2] = ekf.wrapTo180(error_[2] * 180 / np.pi)
            print('>>> Error-xyt: (m/deg) ', error_)

            print('one-iter: ', time.time() - t0, ' sec.')

        # average
        loss_avg = total_loss / pose_cnt

        optimizer.zero_grad()

        loss_avg.backward()

        optimizer.step()

        del ekf

        # writer
        # add loss and error to board
        writer.add_scalar(date+'/Loss', loss_avg, epoch)

        # learning rate reduction
        if epoch % 1 == 0:
            l_rate*=decay
            optimizer = torch.optim.Adam(list(siamese_model.parameters()) + list(patch_model.parameters()), lr=l_rate, weight_decay=w_decay)

    print('Finished')
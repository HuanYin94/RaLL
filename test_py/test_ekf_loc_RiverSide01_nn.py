from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import sys
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
from loader.data_gener_ekf   import dataGener
from network.patch_net_new import patch_net
from network.siamese_triunet_new import siamese_minus
from ekf_filter.ekf_filter_nn import ekf_filter

# multiple radar scans as one real batch
img_batch = 343 # constant to dataset
pose_num = 2215
# pose_num = 7658
# pose_num = 7636
# index from zero, but do not start_id = 0
start_id = 1
end_id = 2215
# end_id = 7658
# end_id = 7636

# d_xyt = np.array([[-3, -2, -1, 0, 1, 2, 3],
#                   [-3, -2, -1, 0, 1, 2, 3],
#                   [-3, -2, -1, 0, 1, 2, 3],])


d_xyt = np.array([[-6, -4, -2, 0, 2, 4, 6],
                  [-6, -4, -2, 0, 2, 4, 6],
                  [-6, -4, -2, 0, 2, 4, 6]])

img_res = 0.25
xySizes = [-1100, 1100, -400, 1800]
radar_size = 512
# # Riverside01
pose_txt = '/home/yinhuan/Data/183/ablation_study/25_512/Riverside01/pose_riverside_01.txt'
map_file = '/home/yinhuan/Data/183/ablation_study/25_512/laser_map_riverside.png'
radar_dir = '/home/yinhuan/Data/183/ablation_study/25_512/Riverside01/radar_scans/'

# odom_file = '/home/yinhuan/Data/radar_loc/05.21/Riverside01_odom.txt'
odom_file = '/home/yinhuan/Data/183/odom/Riverside01_odom.txt'

P_portion = 7

K_portion = 16

alpha = 5

lamda = 1

pre_sia_model_path = '/home/yinhuan/NFS/radar_lite/models/07.10_MCL_25_512_777_r7/siamese_model_200.pth'
pre_patch_model_path = '/home/yinhuan/NFS/radar_lite/models/07.10_MCL_25_512_777_r7/patch_model_200.pth'

# pre_sia_model_path = '/home/yinhuan/Data/183/radar_lite/models/08.17_ekf_nn_upgrade/siamese_model_500.pth'
# pre_patch_model_path = '/home/yinhuan/Data/183/radar_lite/models/08.17_ekf_nn_upgrade/patch_model_500.pth'

# pre_sia_model_path = '/home/yinhuan/NFS/radar_lite/models/07.10_MCL_25_512_set4_r2/siamese_model_300.pth'
# pre_patch_model_path = '/home/yinhuan/NFS/radar_lite/models/07.10_MCL_25_512_set4_r2/patch_model_300.pth'

date = '08.15_ekf_test'

save_est_txt = '/home/yinhuan/Data/183/radar_lite/test/' + date + '/est_riverside_01.txt'

# ekf filter
R = np.diag([1.0,
             1.0,
             np.deg2rad(5.0)]) ** 2 # motion cov

if __name__ == "__main__":

    # set print
    np.set_printoptions(precision=5)

    device = 'cuda:1'
    # device = torch.cuda.current_device()

    data_gener = dataGener(pose_txt, map_file, radar_dir, pose_num, d_xyt,\
         img_res, xySizes, radar_size, img_batch, P_portion, device, odom_file)

    if not os.path.exists('/home/yinhuan/Data/183/radar_lite/test/' + date):
        os.mkdir('/home/yinhuan/Data/183/radar_lite/test/' + date)

    print('Load Model')

    # CAN try multiple GPU
    siamese_model = siamese_minus().to(device)
    siamese_model.load_state_dict(torch.load(pre_sia_model_path))
    # eval only
    siamese_model.eval()
    for param in siamese_model.parameters():
        param.requires_grad = False
    for m in siamese_model.modules():
        if isinstance(m, nn.InstanceNorm2d):
            m.track_running_stats=False

    # create patch model
    patch_model = patch_net(P_portion, K_portion).to(device)
    patch_model.load_state_dict(torch.load(pre_patch_model_path))
    # eval only
    patch_model.eval()
    for param in patch_model.parameters():
        param.requires_grad = False
    for m in patch_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats=False

    # init is needed
    # init at the previous pose since the next step is moion model
    init_id = start_id-1
    pose_init = data_gener.pose[init_id,:]

    # init ekf localization 
    ekf = ekf_filter(pose_init, R, P_portion, d_xyt, lamda, device)

    # write file
    f = open(save_est_txt, 'w+')
    # save the init too
    pose_init_cpu = pose_init.cpu().detach().numpy()
    f.write(str(init_id) + ' ' + str(pose_init_cpu[0]) + ' ' + str(pose_init_cpu[1]) + ' ' + str(pose_init_cpu[2]) + '\n')

    for iter_ in range(start_id, end_id):
        print('------------------------------------------------ ', iter_)

        t0 = time.time()

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

        # print
        g = data_gener.pose[iter_,:].view(-1).cpu().detach().numpy()
        e = est_xyt.view(-1).cpu().detach().numpy()
        print('gt: ', g, ' est: ', e)
        error_ = (g - e)
        error_[2] = ekf.wrapTo180(error_[2] * 180 / np.pi)
        print('>>> Error-xyt: (m/deg) ', error_)
        
        # print
        print('one-iter: ', time.time() - t0, ' sec.')

        # save pose
        f.write(str(iter_) + ' ' + str(e[0]) + ' ' + str(e[1]) + ' ' + str(e[2]) + '\n')

    print('Finished')
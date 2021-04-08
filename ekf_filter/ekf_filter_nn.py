import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import random
import copy

# All in torch tensor

class ekf_filter(torch.nn.Module):
    def __init__(self, pose_gt, R, P_portion, d_xyt, lamda, device):
        super(ekf_filter, self).__init__()
        
        # init pose by gt at first index
        self.R = torch.from_numpy(R).to(device, dtype=torch.float)
        self.est_xyt = pose_gt.reshape(3,1).to(device, dtype=torch.float)
        self.est_xyt_pred = self.est_xyt
        self.P_cov = torch.eye(3).to(device, dtype=torch.float)
        self.P_cov_pred = self.P_cov
        self.P_portion = P_portion
        self.lamda = lamda
        self.device = device

        # hard encode
        self.x_cube = torch.from_numpy(d_xyt[0,:]).reshape(1,P_portion).to(device, dtype=torch.float)
        self.y_cube = torch.from_numpy(d_xyt[1,:]).reshape(1,P_portion).to(device, dtype=torch.float)
        self.t_cube = torch.from_numpy(d_xyt[2,:]).reshape(1,P_portion).to(device, dtype=torch.float) # deg

        # to rad
        self.t_cube_rad = self.deg2rad(self.t_cube)

    def forward(self, gt_pose, avg_vector):
        # reshape
        gt_pose = gt_pose.reshape(3,1)
        
        # get the delta Observation 
        mean_xyt, var_xyt = self.getDeltaMeanVar(avg_vector) # on rad
        
        Z = self.getGlobalObservation(mean_xyt).to(self.device)
    
        # P_z = np.diag([0.5,
        #      0.5,
        #      np.deg2rad(1.0)]) ** 2 # cov

        P_z = torch.diag(var_xyt).to(self.device, dtype=torch.float) # cov
        
        Jacob_H = self.getJacobianH().to(self.device)
        
        S = torch.mm(torch.mm(Jacob_H, self.P_cov_pred), Jacob_H.t()) + P_z
        S = S.to(self.device, dtype=torch.float)
        
        K = torch.mm(torch.mm(self.P_cov_pred, Jacob_H.t()), torch.inverse(S)).to(self.device, dtype=torch.float)
        
        # print('Pred_xyt: ', self.est_xyt_pred)
        # print('Global-Z: ', Z)
        # print('P_z: ')
        # print(P_z)
        # print('K_gain: ')
        # print(K)

        Delta = (Z - self.est_xyt_pred).to(self.device, dtype=torch.float)
        Delta[2] = self.wrapToPi(Delta[2])
        
        # print('Delta: ', Delta)
        # print('K@(Z-X): ', K @ Delta)

        self.est_xyt = self.est_xyt_pred + torch.mm(K, Delta)
        self.est_xyt[2] = self.wrapToPi(self.est_xyt[2])
        
        Iden = torch.eye(3).to(self.device, dtype=torch.float)

        self.P_cov = torch.mm((Iden - torch.mm(K,Jacob_H)), self.P_cov_pred).to(self.device, dtype=torch.float)
        
        #
        # LOSS
        # error_loss 
        error = (gt_pose - self.est_xyt).reshape(3,1).to(self.device, dtype=torch.float)
        error[2] = self.wrapToPi(error[2]) # deBug
        error_loss = torch.mm(torch.mm(error.t(), torch.inverse(self.P_cov)), error)
        
        # cov_loss
        cov_loss = torch.det(self.P_cov)
        
        loss = error_loss + self.lamda * cov_loss

        print('Error-loss: ', error_loss, ' Cov-loss: ', cov_loss, ' Loss: ', loss)

        return loss, self.est_xyt

    def doMotion(self, motion):
        # previous
        # x_ = copy.deepcopy(self.est_xyt)
        # motion ++
        motion = motion.reshape(3,1)
        motion[2] = self.wrapToPi(motion[2])

        motionR = torch.tensor([[math.cos(self.est_xyt[2]), -math.sin(self.est_xyt[2]), 0],
                                [math.sin(self.est_xyt[2]), math.cos(self.est_xyt[2]), 0],
                                [0,0,1]], dtype=torch.float).to(self.device)

        motion_world = torch.mm(motionR, motion)
        # add
        self.est_xyt_pred = self.est_xyt + motion_world
        self.est_xyt_pred[2] = self.wrapToPi(self.est_xyt_pred[2]) # [2] = [2,0]

        # Jacob
        Jacob_F = self.getJacobianF(motion).to(self.device)
        # cov
        self.P_cov_pred = torch.mm(torch.mm(Jacob_F,self.P_cov),Jacob_F.t()) + self.R
        
        # print('Jacob_F: ')
        # print(Jacob_F)
        # print('P_pred: ')
        # print(self.P_cov_pred)
        # print('!!!!!!!!!!!!!!!!!!!!!!')

    def getJacobianF(self, motion):
        # to numpy as temp
        x_ = self.est_xyt.view(-1).cpu().detach().numpy()
        motion = motion.view(-1).cpu().detach().numpy()

        a = -math.sin(x_[2])*motion[0]-math.cos(x_[2])*motion[1]
        b = math.cos(x_[2])*motion[0]-math.sin(x_[2])*motion[1]

        Jacob_F = np.array([[1,0,a],
                            [0,1,b],
                            [0,0,1]], dtype=float)

        return torch.from_numpy(Jacob_F).to(self.device, dtype=torch.float)

    def getDeltaMeanVar(self, avg_vector):
        # inverse as similarity-probability
        avg_vector = -1 * avg_vector

        # P * P * P
        dp_cubes = avg_vector.reshape(self.P_portion, self.P_portion, self.P_portion)
        dp_cubes_sm = dp_cubes.view(-1).softmax(0).view(*dp_cubes.shape)

        # achieve mean_d_xyt
        mean_d_xyt = torch.cat([torch.mm(self.x_cube, dp_cubes_sm.sum(dim=(1,2)).reshape(self.P_portion,1)),\
                            torch.mm(self.y_cube, dp_cubes_sm.sum(dim=(0,2)).reshape(self.P_portion,1)),\
                            torch.mm(self.t_cube, dp_cubes_sm.sum(dim=(0,1)).reshape(self.P_portion,1))], dim=0).view(-1)
        # to rad
        mean_d_xyt[2] = self.deg2rad(mean_d_xyt[2])

        # achieve variance
        # var_d_xyt = torch.zeros(3).to(self.device, dtype=torch.float)
        # prob_x = dp_cubes_sm.sum(dim=(1,2)).reshape(self.P_portion)
        # prob_y = dp_cubes_sm.sum(dim=(0,2)).reshape(self.P_portion)
        # prob_t = dp_cubes_sm.sum(dim=(0,1)).reshape(self.P_portion)
        # for i in range(self.P_portion):
        #     var_d_xyt[0] = var_d_xyt[0] + prob_x[i] * torch.pow((self.x_cube[0,i] - mean_d_xyt[0]),2)
        #     var_d_xyt[1] = var_d_xyt[1] + prob_y[i] * torch.pow((self.y_cube[0,i] - mean_d_xyt[1]),2)
        #     var_d_xyt[2] = var_d_xyt[2] + prob_t[i] * torch.pow((self.t_cube_rad[0,i] - mean_d_xyt[2]),2)

        prob_x = dp_cubes_sm.sum(dim=(1,2)).reshape(self.P_portion,1)
        prob_y = dp_cubes_sm.sum(dim=(0,2)).reshape(self.P_portion,1)
        prob_t = dp_cubes_sm.sum(dim=(0,1)).reshape(self.P_portion,1)

        x_repeat = mean_d_xyt[0].repeat(1, self.P_portion)
        y_repeat = mean_d_xyt[1].repeat(1, self.P_portion)
        t_repeat = mean_d_xyt[2].repeat(1, self.P_portion)

        x_var = torch.mm((self.x_cube-x_repeat), torch.mul(prob_x, (self.x_cube-x_repeat).t()))
        y_var = torch.mm((self.y_cube-y_repeat), torch.mul(prob_y, (self.y_cube-y_repeat).t()))
        t_var = torch.mm((self.t_cube_rad-t_repeat), torch.mul(prob_t, (self.t_cube_rad-t_repeat).t()))

        var_d_xyt = torch.tensor([x_var,y_var,t_var]).view(-1)

        return mean_d_xyt, var_d_xyt # rad for theta

    def getGlobalObservation(self, mean_xyt):
        # from robot-coord to world-coord
        # GPS-like positional + orientational
        Z = torch.zeros(3).to(self.device, dtype=torch.float)
        Z[2] = self.wrapToPi(self.est_xyt_pred[2] + mean_xyt[2]) # rad
        Z[0] = self.est_xyt_pred[0] + math.cos(Z[2])*mean_xyt[0] - math.sin(Z[2])*mean_xyt[1]
        Z[1] = self.est_xyt_pred[1] + math.sin(Z[2])*mean_xyt[0] + math.cos(Z[2])*mean_xyt[1]

        return Z.reshape(3,1)

    def getJacobianH(self):
        Jacob_H = torch.eye(3)
        return Jacob_H.to(self.device, dtype=torch.float)

    def deg2rad(self, deg):
        return deg * np.pi / 180

    def wrapTo180(self, deg):
        deg_new = deg
        while deg_new <= -180:
            deg_new += 360
        while deg_new > 180:
            deg_new -= 360
        return deg_new

    def wrapToPi(self, deg):
        deg_new = deg
        while deg_new <= -np.pi:
            deg_new += np.pi*2
        while deg_new > np.pi:
            deg_new -= np.pi*2
        return deg_new
        
import os
import torch
import torch.nn as nn

class offset_loss(torch.nn.Module):

    def __init__(self, P_portion, alpha, beta, d_xyt, device):
        super(offset_loss, self).__init__()

        self.alpha = alpha
        self.device = device
        self.P_portion = P_portion
        self.beta = beta

        self.x_kedu = d_xyt[0,1] - d_xyt[0,0]
        self.y_kedu = d_xyt[1,1] - d_xyt[1,0]
        self.t_kedu = d_xyt[2,1] - d_xyt[2,0]

        self.x_cube = torch.from_numpy(d_xyt[0,:]).reshape(1,P_portion).to(device, dtype=torch.float)
        self.y_cube = torch.from_numpy(d_xyt[1,:]).reshape(1,P_portion).to(device, dtype=torch.float)
        self.t_cube = torch.from_numpy(d_xyt[2,:]).reshape(1,P_portion).to(device, dtype=torch.float)

        # cross entropy
        self.CELoss = nn.CrossEntropyLoss(reduction='sum').to(device)

        # NLL Loss
        self.NLLLoss = nn.NLLLoss(reduction='sum').to(device)

        # softmin
        self.softmin = nn.Softmin()

    def forward(self, gt_xyt, avg_vector):

        # max
        avg_vector = self.softmin(avg_vector)

        # from 1-d vector to P*P*P cubes
        dp_cubes = avg_vector.reshape(self.P_portion, self.P_portion, self.P_portion)

        # achieve xyt
        est_xyt = torch.cat([torch.mm(self.x_cube.float(), dp_cubes.sum(dim=(1,2)).reshape(self.P_portion,1)),\
                            torch.mm(self.y_cube.float(), dp_cubes.sum(dim=(0,2)).reshape(self.P_portion,1)),\
                            torch.mm(self.t_cube.float(), dp_cubes.sum(dim=(0,1)).reshape(self.P_portion,1))], dim=0).view(-1)

        diff = gt_xyt - est_xyt

        offset_loss = self.alpha * (torch.pow(torch.abs(diff[0]), 2) + \
                            torch.pow(torch.abs(diff[1]), 2)) + \
                            torch.pow(torch.abs(diff[2]), 2)



        # CrossEntropyLoss = Softmax + Log + NLLLoss from pytorch
        # input probability along axis-x-y-t
        input_tensor = torch.cat([dp_cubes.sum(dim=(1,2)).reshape(self.P_portion,1),\
                                    dp_cubes.sum(dim=(0,2)).reshape(self.P_portion,1),\
                                    dp_cubes.sum(dim=(0,1)).reshape(self.P_portion,1)], dim=0).reshape(3,self.P_portion)
        # make sure non-zero in the probabilities for log-operation
        input_tensor = torch.clamp(input_tensor, min=1e-8)
        # log
        input_tensor = torch.log(input_tensor)

        # target id tensor from ground truth
        delta_x = torch.abs(torch.sub(gt_xyt[0].repeat(1,self.P_portion), self.x_cube))
        delta_y = torch.abs(torch.sub(gt_xyt[1].repeat(1,self.P_portion), self.y_cube))
        delta_t = torch.abs(torch.sub(gt_xyt[2].repeat(1,self.P_portion), self.t_cube))

        gt_id_x = (delta_x == torch.min(delta_x)).nonzero()
        gt_id_y = (delta_y == torch.min(delta_y)).nonzero()
        gt_id_t = (delta_t == torch.min(delta_t)).nonzero()

        target_id_tensor = torch.tensor([gt_id_x[0,1], gt_id_y[0,1], gt_id_t[0,1]]).reshape(3).to(self.device)

        class_loss = self.NLLLoss(input_tensor, target_id_tensor)

        # merge
        # print('offset_loss: ', offset_loss.detach().cpu().numpy(), ' class_loss: ', class_loss.detach().cpu().numpy())
        loss = offset_loss + self.beta * class_loss

        return loss, est_xyt

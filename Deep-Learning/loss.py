import torch
import torch.nn.functional as F
import torch.nn as nn
# def rmse_loss(output, target):
#     mse = F.mse_loss(output, target)
#     rmse = torch.sqrt(mse)
#     return rmse



class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
    
    def forward(self,output,target):
        mse = F.mse_loss(output, target)
        rmse = torch.sqrt(mse)
        return rmse
import torch
import torch.nn.functional as F

def rmse_loss(output, target):
    mse = F.mse_loss(output, target)
    rmse = torch.sqrt(mse)
    return rmse

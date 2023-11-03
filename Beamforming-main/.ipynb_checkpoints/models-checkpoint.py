import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
##Model
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.linear1 = nn.Linear(10, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 512)
        self.linear5 = nn.Linear(512, 256)
        self.linear6 = nn.Linear(256, 128)
        self.linear7 = nn.Linear(128, 128)  

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        x = self.linear7(x)
        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.linear1 = nn.Linear(10, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 512)
        self.linear5 = nn.Linear(512, 256)
        self.linear6 = nn.Linear(256, 128)
        self.linear7 = nn.Linear(128, 128)  

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
        x = F.tanh(self.linear5(x))
        x = F.sigmoid(self.linear6(x))
        x = self.linear7(x)
        return x


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.Linear1 = nn.Linear(10, 64)
        self.Linear2 = nn.Linear(64, 128)
        self.Linear3 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.tanh(self.Linear1(x))
        x = F.tanh(self.Linear2(x))
        x = F.sigmoid(self.Linear3(x))
        return x

